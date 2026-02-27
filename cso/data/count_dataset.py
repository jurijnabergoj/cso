from torch.utils.data import Dataset
import numpy as np
import torch
from pathlib import Path


class CountDataset(Dataset):
    """
    Dataset for counting the "Stacks-3D-Real" dataset.
    Assumes data structure:
    scenes/
        beads/
            embeddings.pt
        beads2/
            ...
    """

    def __init__(self, data_dirs):
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]
        
        self.data_dirs = [Path(d) for d in data_dirs]
        self.samples = []
        for dir in self.data_dirs:
            self.samples.extend([
                d for d in dir.iterdir()
                if (d / "embeddings.pt").exists()
            ])
        self.samples = sorted(self.samples)
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = torch.load(self.samples[idx] / "embeddings.pt", map_location="cpu")
        
        return {
            "container_outputs": data["container"],
            "object_outputs": data["object"],
            "true_count": torch.tensor(
                data["true_count"], dtype=torch.float32
            ),
            "geom_features": data["geom_features"],
            "geom_estimate": torch.tensor(
                data["geom_estimate"], dtype=torch.float32
            ),
            "sample_name": self.samples[idx].name,
        }


def collate_fn(batch):
    return {
        "container_outputs": {
            "coords": [b["container_outputs"]["coords"] for b in batch],
            "slat_features": [
                b["container_outputs"]["slat_features"] for b in batch
            ],
            "shape_latent": torch.stack([
                b["container_outputs"]["shape_latent"] for b in batch
            ]),
            "scale": torch.stack([
                b["container_outputs"]["scale"] for b in batch
            ]),
            "translation": torch.stack([
                b["container_outputs"]["translation"] for b in batch
            ]),
            "rotation": torch.stack([
                b["container_outputs"]["rotation"] for b in batch
            ]),
        },

        "object_outputs": {
            "coords": [b["object_outputs"]["coords"] for b in batch],
            "slat_features": [
                b["object_outputs"]["slat_features"] for b in batch
            ],
            "shape_latent": torch.stack([
                b["object_outputs"]["shape_latent"] for b in batch
            ]),
            "scale": torch.stack([
                b["object_outputs"]["scale"] for b in batch
            ]),
            "translation": torch.stack([
                b["object_outputs"]["translation"] for b in batch
            ]),
            "rotation": torch.stack([
                b["object_outputs"]["rotation"] for b in batch
            ]),
        },

        "true_count": torch.stack([b["true_count"] for b in batch]),
        "geom_features": torch.stack([b["geom_features"] for b in batch]),
        "geom_estimate": torch.stack([b["geom_estimate"] for b in batch]),
        "sample_name": [b["sample_name"] for b in batch],
    }