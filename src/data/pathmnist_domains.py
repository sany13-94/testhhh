# src/data/pathmnist_domains.py
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np

# ---------- Augment & domain-shift ----------

def build_augmentation_transform():  
    t = []
    t.append(transforms.RandomHorizontalFlip(p=0.5))
    t.append(transforms.RandomRotation(degrees=45))
    t.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5))
    return transforms.Compose(t)

class SameModalityDomainShift:
    def __init__(self, domain_id: int, seed: int = 42):
        self.domain_id = domain_id
        self.seed = seed
        self.characteristics = self._generate_domain_characteristics()
        
    def _generate_domain_characteristics(self):
        profiles = {
            0: {'name':'high_end',  'noise_level':0.00, 'contrast_scale':1.0, 'brightness_shift':0.0},
            1: {'name':'mid_range', 'noise_level':0.08, 'contrast_scale':0.75, 'brightness_shift':0.15},
            2: {'name':'older_model','noise_level':0.15,'contrast_scale':0.60,'brightness_shift':0.20},
            3: {'name':'very_old',  'noise_level':0.20,'contrast_scale':0.55,'brightness_shift':0.25},  # optional 4th domain
        }
        p = profiles.get(self.domain_id, profiles[0])
        return dict(
            name=p['name'],
            noise_level=p['noise_level'],
            contrast_scale=p['contrast_scale'],
            brightness_shift=p['brightness_shift'],
        )
    
    def apply_transform(self, img: torch.Tensor) -> torch.Tensor:
        # img float tensor in [0,1], shape (C,H,W)
        if self.domain_id == 0:
            return img
        img = img * self.characteristics['contrast_scale']
        img = img + self.characteristics['brightness_shift']
        if self.characteristics['noise_level'] > 0:
            img = img + torch.randn_like(img) * self.characteristics['noise_level']
        return img.clamp(0.0, 1.0)

class DomainShiftedPathMNIST(Dataset):
    def __init__(self, base_ds: Dataset, domain_id: int, seed: int = 42):
        self.base = base_ds
        self.shift = SameModalityDomainShift(domain_id=domain_id, seed=seed)

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        img = self.shift.apply_transform(img)
        return img, label

class LazyPathMNIST(Dataset):
    """Loads PathMNIST and returns tensors in [0,1] (no augment, no domain shift)."""
    def __init__(self, split: str):
        import medmnist
        self.to_tensor = transforms.ToTensor()
        if split not in ['train', 'test', 'val']:
            raise ValueError("split must be 'train' or 'test' or 'val'")
        ds_class = getattr(medmnist.dataset, 'PathMNIST')
        self.ds = ds_class(split=split, transform=None, download=True)
        self.imgs = self.ds.imgs                                  # (N, H, W, C)
        self.labels = self.ds.labels.flatten().astype(np.int64)   # (N,)

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx].copy()              # HWC uint8
        img_t = self.to_tensor(img)              # CHW float in [0,1]
        label = int(self.labels[idx])
        return img_t, torch.tensor(label, dtype=torch.long)

class AugmentationWrapper(Dataset):
    """Apply augmentation AFTER domain shift (train only)."""
    def __init__(self, base_ds: Dataset, augmentation: transforms.Compose):
        self.base = base_ds
        self.augmentation = augmentation
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.augmentation(img), label

# ---------- Build per-client loaders ----------

def _partition_list_lengths(n_items: int, n_parts: int):
    base = n_items // n_parts
    rem  = n_items % n_parts
    return [base + 1] * rem + [base] * (n_parts - rem)

def make_pathmnist_clients_with_domains(
    k: int = 15,        # total train clients
    d: int = 4,         # number of domains
    batch_size: int = 32,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    aug = build_augmentation_transform()
    ds_train = LazyPathMNIST(split='train')
    ds_test  = LazyPathMNIST(split='test')

    # assign clients to domains evenly
    domain_assignment = []
    per_dom = k // d
    for dom in range(d):
        domain_assignment += [dom] * per_dom
    while len(domain_assignment) < k:
        domain_assignment.append(len(domain_assignment) % d)

    # partition training set into k shards
    g = torch.Generator().manual_seed(seed)
    lengths = _partition_list_lengths(len(ds_train), k)
    shards = random_split(ds_train, lengths, generator=g)

    train_loaders, val_loaders = [], []
    for cid in range(k):
        shard = shards[cid]
        n_val = max(1, int(len(shard) * val_ratio))
        n_trn = len(shard) - n_val
        g2 = torch.Generator().manual_seed(seed + cid)
        trn_base, val_base = random_split(shard, [n_trn, n_val], generator=g2)

        dom_id = domain_assignment[cid]
        trn_shift = DomainShiftedPathMNIST(trn_base, dom_id, seed)
        val_shift = DomainShiftedPathMNIST(val_base, dom_id, seed)
        trn_aug   = AugmentationWrapper(trn_shift, aug)

        train_loaders.append(DataLoader(trn_aug, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True))
        val_loaders.append(  DataLoader(val_shift, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True))

    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    train_sizes = [len(dl.dataset) for dl in train_loaders]  # used by FedAvg & Pow-d

    return train_loaders, val_loaders, test_loader, domain_assignment, train_sizes

# ---------- Adapter class expected by your main.py ----------

class PathMNISTDomainDataset:
    """
    Exposes the same attributes/shapes your main.py expects:
      - num_classes
      - train_num_clients
      - test_num_clients
      - dataset: dict with ['train']['dataloader_list'], ['train']['data_sizes'], ['test']...
    """
    def __init__(self, args):
        # read knobs (use defaults if not present)
        k = getattr(args, "num_total_clients", None) or getattr(args, "total_num_client", None) or 15
        d = getattr(args, "domains", 4)
        batch = getattr(args, "batch_size", 32)
        val_ratio = getattr(args, "val_ratio", 0.1)
        seed = getattr(args, "seed", 42)

        # build loaders FIRST
        (train_loaders,
         val_loaders,
         test_loader,
         domain_assignment,
         train_sizes) = make_pathmnist_clients_with_domains(
            k=k, d=d, batch_size=batch, val_ratio=val_ratio, seed=seed
        )

        # now set them on self
        self.train_loaders      = train_loaders
        self.val_loaders        = val_loaders
        self.test_loader        = test_loader
        self.domain_assignment  = domain_assignment
        self.train_sizes        = train_sizes

        # meta
        self.num_classes = 9                       # PathMNIST has 9 classes
        self.train_num_clients = len(self.train_loaders)
        self.test_num_clients  = 1

        # the dict that your main.py / Server expect
        self.dataset = {
            'train': {
                'dataloader_list': self.train_loaders,
                'data_sizes':      self.train_sizes,
                'domain_assignment': self.domain_assignment,  # for logging, optional
                'val_dataloader_list': self.val_loaders       # if your Server uses it
            },
            'test': {
                'dataloader_list': [self.test_loader],
                'data_sizes':      [len(self.test_loader.dataset)]
            }
        }
