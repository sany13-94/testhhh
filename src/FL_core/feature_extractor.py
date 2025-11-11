# src/FL_core/feature_extractor.py
import torch
import torch.nn as nn
import numpy as np

class FeatureExtractor(nn.Module):
    """
    Wrap a classifier and return penultimate features.
    Works for ResNet-like models (conv trunk -> avgpool -> fc).
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            # up to avgpool (exclude the final fc)
            self.backbone = nn.Sequential(*(list(model.children())[:-1]))
            self.feat_dim = model.fc.in_features
        else:
            raise RuntimeError(
                "FeatureExtractor: please adapt for your model type (no .fc found)."
            )

    def forward(self, x):
        z = self.backbone(x)       # (B, C, 1, 1)
        z = torch.flatten(z, 1)    # (B, C)
        return z


@torch.no_grad()
def compute_macro_prototype_from_loader(dataloader, feature_net, device, max_batches=3):
    """
    Compute a macro prototype: average of per-class mean features.
    Uses up to max_batches batches to reduce cost.
    """
    feature_net.eval()
    class_sums, class_counts = {}, {}
    n_batches = 0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        z = feature_net(x)  # [B, D]
        for c in y.unique().tolist():
            mask = (y == c)
            if mask.any():
                cls_vec = z[mask].mean(dim=0)  # mean feature for class c in this batch
                class_sums[c]  = class_sums.get(c, 0) + cls_vec
                class_counts[c] = class_counts.get(c, 0) + 1
        n_batches += 1
        if n_batches >= max_batches:
            break

    if not class_sums:
        return None
    per_class = []
    for c, s in class_sums.items():
        per_class.append((s / class_counts[c]).unsqueeze(0))
    macro = torch.cat(per_class, dim=0).mean(dim=0)  # [D]
    return macro.detach().cpu().numpy()


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b) / denom)
