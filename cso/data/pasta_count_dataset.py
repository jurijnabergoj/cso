from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torch


class PastaCountDataset(Dataset):
    """
    Dataset for pasta counting.
    Assumes data structure:
    data_dir/
        sample_001/
            image.png
            container_mask.png
            object_mask.png
            count.txt
        sample_002/
            ...
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.samples = sorted([
            d for d in self.data_dir.iterdir()
            if (d / "embeddings.pt").exists()
        ])
        
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
            "sample_name": self.samples[idx].name,
        }


def collate_fn(batch):
    return batch
