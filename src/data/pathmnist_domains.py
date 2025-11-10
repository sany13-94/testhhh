from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split
from torchvision import transforms

# ------------------ Augmentations ------------------

def build_augmentation_transform():
    t = []
    t.append(transforms.RandomHorizontalFlip(p=0.5))
    t.append(transforms.RandomRotation(degrees=45))
    t.append(transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5))
    return transforms.Compose(t)

# ------------------ Domain Shift ------------------

class SameModalityDomainShift:
    """
    Domain 0: identity
    Domain 1: brightness in [-0.2, +0.15], contrast in [0.6, 1.4], reduced resolution
    Domain 2: brightness +0.3, noise sigma=0.12, contrast in [0.5, 1.5]
    Domain 3: identity (test-domain client built from test split)
    """
    def __init__(self, domain_id: int, seed: int = 42):
        self.domain_id = domain_id
        self.rng = np.random.default_rng(seed)

    def _rand(self, a: float, b: float) -> float:
        return float(self.rng.uniform(a, b))

    def _reduce_resolution(self, img: torch.Tensor, scale=0.7) -> torch.Tensor:
        # img: (C,H,W), float in [0,1]
        C, H, W = img.shape
        new_h, new_w = max(1, int(H * scale)), max(1, int(W * scale))
        img_small = F.interpolate(img.unsqueeze(0), size=(new_h, new_w),
                                  mode="bilinear", align_corners=False)
        img_back = F.interpolate(img_small, size=(H, W),
                                 mode="bilinear", align_corners=False)
        return img_back.squeeze(0).clamp(0.0, 1.0)

    def apply(self, img: torch.Tensor) -> torch.Tensor:
        if self.domain_id == 0:
            return img

        if self.domain_id == 1:
            # modest brightness, contrast, and resolution drop
            b = self._rand(-0.2, 0.15)
            c = self._rand(0.6, 1.4)
            out = img * c + b
            out = self._reduce_resolution(out, scale=0.7)
            return out.clamp(0.0, 1.0)

        if self.domain_id == 2:
            # degraded: fixed +0.3 brightness, noise, wider contrast jitter
            c = self._rand(0.5, 1.5)
            out = img * c + 0.3
            noise = torch.randn_like(out) * 0.12
            out = out + noise
            return out.clamp(0.0, 1.0)

        # domain 3 (test-domain) -> identity by default
        return img

class DomainShiftedWrapper(Dataset):
    """Wrap base dataset and apply SameModalityDomainShift per sample."""
    def __init__(self, base: Dataset, domain_id: int, seed: int = 42):
        self.base = base
        self.shift = SameModalityDomainShift(domain_id, seed)

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        img, y = self.base[idx]
        return self.shift.apply(img), y

class AugmentationWrapper(Dataset):
    """Apply augmentations after domain shift (for training only)."""
    def __init__(self, base: Dataset, aug: transforms.Compose):
        self.base = base
        self.aug = aug
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, y = self.base[idx]
        return self.aug(img), y

# ------------------ Raw PathMNIST loader (no transforms) ------------------

class LazyPathMNIST(Dataset):
    """Return (C,H,W) tensors in [0,1]; no aug/shift here."""
    def __init__(self, split: str):
        import medmnist
        self.to_tensor = transforms.ToTensor()
        if split not in ['train', 'test', 'val']:
            raise ValueError("split must be 'train'|'test'|'val'")
        cls = getattr(medmnist.dataset, 'PathMNIST')
        ds = cls(split=split, transform=None, download=True)
        self.imgs = ds.imgs                                  # (N,H,W,C)
        self.labels = ds.labels.flatten().astype(np.int64)   # (N,)

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx].copy()              # HWC uint8
        img_t = self.to_tensor(img)              # CHW float [0,1]
        y = int(self.labels[idx])
        return img_t, torch.tensor(y, dtype=torch.long)

# ------------------ Build EMNIST-style dict ------------------

def _even_partition_lengths(n_items: int, n_parts: int):
    base = n_items // n_parts
    rem  = n_items % n_parts
    return [base + 1] * rem + [base] * (n_parts - rem)

def build_pathmnist_emnist_style(
    k: int = 15,           # total clients INCLUDING the test-domain client
    d: int = 3,            # number of TRAIN domains (0,1,2)
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    - Split TRAIN into K-1 clients across 3 domains (0/1/2).
    - Build ONE extra client from TEST split as domain 3.
    - Return EMNIST-like dict where:
        * dataset['train'] = { 'data_sizes': {cid:n}, 'data': {cid: Dataset-with-aug} }
        * dataset['test']  = { 'data_sizes': {cid:n}, 'data': {cid: VAL-Dataset-no-aug} }
      (We use per-client VAL as "test" so the server's evaluation is your validation.)
    """
    assert d == 3, "This implementation assumes 3 training domains (0,1,2)."

    ds_train = LazyPathMNIST(split='train')
    ds_test  = LazyPathMNIST(split='test')
    aug = build_augmentation_transform()

    ds_train = _trim_to_multiple(ds_train, 32)
    ds_test   = _trim_to_multiple(ds_test,  32)

    num_train_clients = k - 1  # last client will be built from test split (domain 3)

    # ---- Assign train clients to domains 0..2 as in your code ----
    clients_per_domain = num_train_clients // d
    domain_assignment = []
    for dom in range(d):
        domain_assignment += [dom] * clients_per_domain
    # distribute remaining clients evenly
    while len(domain_assignment) < num_train_clients:
        domain_assignment.append(len(domain_assignment) % d)

    # ---- Partition train into (k-1) shards; each has its own val ----
    g = torch.Generator().manual_seed(seed)
    lengths = _even_partition_lengths(len(ds_train), num_train_clients)
    shards  = random_split(ds_train, lengths, generator=g)

    train_data_local_dict, train_sizes = {}, {}
    val_data_local_dict,   val_sizes   = {}, {}

    for cid in range(num_train_clients):
        shard = shards[cid]
        # per-client train/val
        n_val = int(len(shard) * val_ratio)
        n_trn = len(shard) - n_val
        g2 = torch.Generator().manual_seed(seed + cid)
        trn_base, val_base = random_split(shard, [n_trn, n_val], generator=g2)

        dom_id = domain_assignment[cid]

        # domain shift
        tr_shift = DomainShiftedWrapper(trn_base, dom_id, seed)
        va_shift = DomainShiftedWrapper(val_base, dom_id, seed)

        # augment train only
        tr_final = AugmentationWrapper(tr_shift, aug)

        train_data_local_dict[cid] = tr_final
        train_sizes[cid] = len(tr_final)

        val_data_local_dict[cid] = va_shift
        val_sizes[cid] = len(va_shift)

    # ---- Extra client from TEST split = domain 3 ----
    # We mimic your code: create its own train/val from the test split
    n_val_test = int(len(ds_test) * val_ratio)
    n_trn_test = len(ds_test) - n_val_test
    g_test = torch.Generator().manual_seed(seed + k - 1)
    trn_test_base, val_test_base = random_split(ds_test, [n_trn_test, n_val_test], generator=g_test)

    dom_id_test = 3  # fourth domain
    tr_shift_test = DomainShiftedWrapper(trn_test_base, dom_id_test, seed)
    va_shift_test = DomainShiftedWrapper(val_test_base, dom_id_test, seed)
    tr_final_test = AugmentationWrapper(tr_shift_test, aug)

    train_data_local_dict[num_train_clients] = tr_final_test
    train_sizes[num_train_clients] = len(tr_final_test)

    val_data_local_dict[num_train_clients] = va_shift_test
    val_sizes[num_train_clients] = len(va_shift_test)

    # full domain map (K entries): first K-1 from 0..2, last one = 3
    domain_assignment_all = domain_assignment + [3]

    dataset = {
        'train': {
            'data_sizes': train_sizes,          # dict: cid -> int
            'data':       train_data_local_dict # dict: cid -> Dataset (augmented)
        },
        'test': {  # we store VALIDATION sets here so the server evaluates on val
            'data_sizes': val_sizes,
            'data':       val_data_local_dict
        },
        'meta': {
            'domain_assignment': domain_assignment_all,
            'note': 'PathMNIST K-1 train clients across 3 domains (0,1,2) + 1 test-domain client (3).',
        }
    }
    return dataset

# ------------------ Public class used by main.py ------------------
from torch.utils.data import Subset

def _trim_to_multiple(ds, batch_size: int):
    L = len(ds)
    if L < batch_size:
        return ds
    L_trim = (L // batch_size) * batch_size
    if L_trim == L:
        return ds
    return Subset(ds, list(range(L_trim)))
class PathMNISTDomainDataset:
    """Mirror EMNIST interface so the existing Server works unchanged."""
    def __init__(self, args):
        k = getattr(args, 'total_num_client', None) or getattr(args, 'num_total_clients', None) or 15
        d = getattr(args, 'domains', 3)
        seed = getattr(args, 'seed', 42)
        val_ratio = getattr(args, 'val_ratio', 0.1)

        self.dataset = build_pathmnist_emnist_style(k=k, d=d, val_ratio=val_ratio, seed=seed)
        self.num_classes = 9
        self.train_num_clients = len(self.dataset['train']['data_sizes'])
        self.test_num_clients  = len(self.dataset['test']['data_sizes'])