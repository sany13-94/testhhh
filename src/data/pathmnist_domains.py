# src/data/pathmnist_domains.py
from typing import List, Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import numpy as np

# === your augmentation and dataset code (lightly adapted) ===

def build_augmentation_transform():  
    t = []
    t.append(transforms.RandomHorizontalFlip(p=0.5))
    t.append(transforms.RandomRotation(degrees=45))
    t.append(transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.5))
    return transforms.Compose(t)

class SameModalityDomainShift:
    def __init__(self, domain_id: int, modality: str = "CT", seed: int = 42):
        self.domain_id = domain_id
        self.seed = seed
        self.characteristics = self._generate_domain_characteristics()
        
    def _generate_domain_characteristics(self):
        equipment_profiles = {
            0: {'name':'high_end', 'noise_level':0.00, 'contrast_scale':1.0, 'brightness_shift':0.0},
            1: {'name':'mid_range', 'noise_level':0.08, 'contrast_scale':0.75, 'brightness_shift':0.15},
            2: {'name':'older_model','noise_level':0.15,'contrast_scale':0.60,'brightness_shift':0.20},
        }
        p = equipment_profiles.get(self.domain_id, equipment_profiles[0])
        return dict(name=p['name'], noise_level=p['noise_level'],
                    contrast_scale=p['contrast_scale'], brightness_shift=p['brightness_shift'])
    
    def apply_transform(self, img: torch.Tensor) -> torch.Tensor:
        # expecting img in [0,1], float tensor (C,H,W)
        if self.domain_id == 0:
            return img
        img = img * self.characteristics['contrast_scale']
        img = img + self.characteristics['brightness_shift']
        if self.characteristics['noise_level'] > 0:
            noise = torch.randn_like(img) * self.characteristics['noise_level']
            img = img + noise
        # keep range sane for training (avoid extreme values after noise/shift)
        return img.clamp(0.0, 1.0)

class DomainShiftedPathMNIST(Dataset):
    def __init__(self, base_ds: Dataset, domain_id: int, seed: int = 42):
        self.base = base_ds
        self.domain_id = domain_id
        self.shift = SameModalityDomainShift(domain_id=domain_id, seed=seed)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        img = self.shift.apply_transform(img)
        return img, label

class LazyPathMNIST(Dataset):
    """
    Loads PathMNIST and returns tensors in [0,1] (no augment, no domain shift).
    """
    def __init__(self, split: str):
        import medmnist
        self.to_tensor = transforms.ToTensor()
        if split not in ['train', 'test', 'val']:
            raise ValueError("Split must be one of 'train', 'test', or 'val'.")
        dataset_class = getattr(medmnist.dataset, 'PathMNIST')
        self.medmnist_ds = dataset_class(split=split, transform=None, download=True)
        self.imgs = self.medmnist_ds.imgs          # (N, H, W, C)
        self.labels = self.medmnist_ds.labels.flatten().astype(np.int64)  # (N,)

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx].copy()
        img_t = self.to_tensor(img)  # (C,H,W) in [0,1]
        label = int(self.labels[idx])
        return img_t, torch.tensor(label, dtype=torch.long)

class AugmentationWrapper(Dataset):
    def __init__(self, base_ds: Dataset, augmentation: transforms.Compose):
        self.base = base_ds
        self.augmentation = augmentation
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, label = self.base[idx]
        img = self.augmentation(img)
        return img, label

def _get_client_signature(ds: Dataset, n=512):
    # quick histogram for sanity (optional)
    counts = {}
    for i in range(min(n, len(ds))):
        _, y = ds[i]
        y = int(y)
        counts[y] = counts.get(y, 0) + 1
    return counts

def make_pathmnist_clients_with_domains(
    k: int = 15,        # total clients (k-1 train clients + 1 extra "test" client, optional)
    d: int = 3,         # number of domains
    batch_size: int = 32,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    augmentation_transform = build_augmentation_transform()
    ds_train = LazyPathMNIST(split='train')
    ds_test  = LazyPathMNIST(split='test')

    num_train_clients = k  # we will just use k train clients; validation per client; test is global

    # Assign clients to domains
    clients_per_domain = num_train_clients // d
    domain_assignment = []
    for domain_id in range(d):
        domain_assignment += [domain_id] * clients_per_domain
    while len(domain_assignment) < num_train_clients:
        domain_assignment.append(len(domain_assignment) % d)

    # Partition train among clients
    g = torch.Generator().manual_seed(seed)
    base_size = len(ds_train) // num_train_clients
    remainder = len(ds_train) % num_train_clients
    part_lengths = [base_size + 1] * remainder + [base_size] * (num_train_clients - remainder)
    raw_parts = random_split(ds_train, part_lengths, generator=g)

    train_loaders, val_loaders = [], []
    for cid in range(num_train_clients):
        part_ds = raw_parts[cid]
        n_val = max(1, int(len(part_ds) * val_ratio))
        n_trn = len(part_ds) - n_val
        g_split = torch.Generator().manual_seed(seed + cid)
        trn_base, val_base = random_split(part_ds, [n_trn, n_val], generator=g_split)

        dom_id = domain_assignment[cid]

        shifted_trn = DomainShiftedPathMNIST(trn_base, dom_id, seed=seed)
        shifted_val = DomainShiftedPathMNIST(val_base, dom_id, seed=seed)
        aug_trn     = AugmentationWrapper(shifted_trn, augmentation_transform)

        train_loaders.append(DataLoader(aug_trn, batch_size=batch_size, shuffle=True,
                                        num_workers=2, pin_memory=True))
        val_loaders.append(DataLoader(shifted_val, batch_size=batch_size, shuffle=False,
                                      num_workers=2, pin_memory=True))

    # single global test loader (no shift; evaluate domain-agnostic)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # sizes for FedAvg weights, Pow-d setup, etc.
    train_sizes = [len(dl.dataset) for dl in train_loaders]

    # signatures (optional debugging)
    # for i, p in enumerate(raw_parts):
    #     print(f"Client {i} signature:", _get_client_signature(p))

    return train_loaders, val_loaders, test_loader, domain_assignment, train_sizes

# === FL framework adapter ===
class PathMNISTDomainDataset:
    """
    Adapter exposing the attributes most FL codebases expect:
      - train_loaders: List[DataLoader] for each client (local train)
      - val_loaders:   List[DataLoader] for each client (local val)
      - test_loader:   DataLoader (global test)
      - domain_assignment: List[int] aligned with clients
      - train_sizes:   List[int] (used by FedAvg weights, Pow-d setup)
      - num_clients:   int
    """
    def __init__(self, args):
        k = getattr(args, "num_total_clients", None) or getattr(args, "num_clients", None) or 15
        d = getattr(args, "domains", 3)
        batch = getattr(args, "batch_size", 32)
        val_ratio = getattr(args, "val_ratio", 0.1)
        seed = getattr(args, "seed", 42)
        # ---- expose what main.py expects ----
        self.num_classes = 9  # PathMNIST has 9 classes
        self.train_num_clients = len(self.train_loaders)
        self.test_num_clients = 1  # single global test loader

        (self.train_loaders,
         self.val_loaders,
         self.test_loader,
         self.domain_assignment,
         self.train_sizes) = make_pathmnist_clients_with_domains(
            k=k, d=d, batch_size=batch, val_ratio=val_ratio, seed=seed
        )
        self.num_clients = len(self.train_loaders)
        self.dataset = {
            'train': {
                'dataloader_list': self.train_loaders,
                'data_sizes': self.train_sizes,
                'domain_assignment': self.domain_assignment,   # optional but useful
                # (optional) you can store val loaders if your Server uses them:
                'val_dataloader_list': self.val_loaders
            },
            'test': {
                'dataloader_list': [self.test_loader],
                'data_sizes': [len(self.test_loader.dataset)]
            }
        }
