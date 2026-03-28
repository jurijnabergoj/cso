from torch.utils.data import Dataset
import numpy as np
import torch
from pathlib import Path


class CountDataset(Dataset):
    def __init__(
        self,
        data_dirs,
        cache_in_memory=False,
        image_feat_file="image_feats.pt",
        container_image_feat_file="",
        object_image_feat_file="",
    ):
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]

        self.data_dirs = [Path(d) for d in data_dirs]
        self.image_feat_file = image_feat_file
        self.container_image_feat_file = container_image_feat_file
        self.object_image_feat_file = object_image_feat_file

        self.samples = []
        for dir in self.data_dirs:
            self.samples.extend(
                [d for d in dir.iterdir() if (d / "embeddings.pt").exists()]
            )
        self.samples = sorted(self.samples)

        self._cache = None
        if cache_in_memory:
            self._cache = [self._load(i) for i in range(len(self.samples))]

    def __len__(self):
        return len(self.samples)

    def _load(self, idx):
        path = self.samples[idx] / "embeddings.pt"
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Corrupted embedding found at path: {path} and index: {idx}")
            raise e

        image_feats_path = self.samples[idx] / self.image_feat_file
        image_feats = (
            torch.load(image_feats_path, map_location="cpu", weights_only=True)
            if image_feats_path.exists()
            else None
        )

        def _load_features(filename):
            if not filename:
                return None
            p = self.samples[idx] / filename
            return (
                torch.load(p, map_location="cpu", weights_only=True)
                if p.exists()
                else None
            )

        container_image_feats = _load_features(self.container_image_feat_file)
        object_image_feats = _load_features(self.object_image_feat_file)

        return {
            "container_outputs": data["container"],
            "object_outputs": data["object"],
            "true_count": torch.tensor(data["true_count"], dtype=torch.float32),
            "geom_features": data["geom_features"],
            "geom_estimate": torch.tensor(data["geom_estimate"], dtype=torch.float32),
            "image_feats": image_feats,
            "container_image_feats": container_image_feats,
            "object_image_feats": object_image_feats,
            "sample_name": self.samples[idx].name,
        }

    def __getitem__(self, idx):
        if self._cache is not None:
            return self._cache[idx]
        return self._load(idx)


def _pool_slat(x):
    """
    Max + mean pool
    (N, D) slat tensor -> (2*D,) vector.
    """
    return torch.cat([x.max(dim=0)[0], x.mean(dim=0)], dim=-1)


def collate_fn(batch):
    return {
        "container_outputs": {
            "coords": [b["container_outputs"]["coords"] for b in batch],
            "slat_features": torch.stack(
                [_pool_slat(b["container_outputs"]["slat_features"]) for b in batch]
            ),
            "shape_latent": torch.stack(
                [b["container_outputs"]["shape_latent"] for b in batch]
            ),
            "scale": torch.stack([b["container_outputs"]["scale"] for b in batch]),
            "translation": torch.stack(
                [b["container_outputs"]["translation"] for b in batch]
            ),
            "rotation": torch.stack(
                [b["container_outputs"]["rotation"] for b in batch]
            ),
        },
        "object_outputs": {
            "coords": [b["object_outputs"]["coords"] for b in batch],
            "slat_features": torch.stack(
                [_pool_slat(b["object_outputs"]["slat_features"]) for b in batch]
            ),
            "shape_latent": torch.stack(
                [b["object_outputs"]["shape_latent"] for b in batch]
            ),
            "scale": torch.stack([b["object_outputs"]["scale"] for b in batch]),
            "translation": torch.stack(
                [b["object_outputs"]["translation"] for b in batch]
            ),
            "rotation": torch.stack([b["object_outputs"]["rotation"] for b in batch]),
        },
        "true_count": torch.stack([b["true_count"] for b in batch]),
        "geom_features": torch.stack([b["geom_features"] for b in batch]),
        "geom_estimate": torch.stack([b["geom_estimate"] for b in batch]),
        "image_feats": (
            torch.stack([b["image_feats"] for b in batch])
            if all(b["image_feats"] is not None for b in batch)
            else None
        ),
        "container_image_feats": (
            torch.stack([b["container_image_feats"] for b in batch])
            if all(b["container_image_feats"] is not None for b in batch)
            else None
        ),
        "object_image_feats": (
            torch.stack([b["object_image_feats"] for b in batch])
            if all(b["object_image_feats"] is not None for b in batch)
            else None
        ),
        "sample_name": [b["sample_name"] for b in batch],
    }
