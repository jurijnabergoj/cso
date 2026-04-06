from torch.utils.data import Dataset
from PIL import Image
import json
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
                [
                    d
                    for d in dir.iterdir()
                    if (d / "sam3d_data" / "embeddings.pt").exists()
                    or (d / "embeddings.pt").exists()
                ]
            )
        self.samples = sorted(self.samples)

        self._cache = None
        if cache_in_memory:
            self._cache = [self._load(i) for i in range(len(self.samples))]

    def __len__(self):
        return len(self.samples)

    def _load(self, idx):
        sam3d_path = self.samples[idx] / "sam3d_data" / "embeddings.pt"
        path = (
            sam3d_path if sam3d_path.exists() else self.samples[idx] / "embeddings.pt"
        )
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
            if not p.exists():
                return None
            feat = torch.load(p, map_location="cpu", weights_only=True)
            # Mean-pool instance features (k, D) -> (D,)
            if feat.dim() == 2:
                feat = feat.float().mean(dim=0)
            return feat

        container_image_feats = _load_features(self.container_image_feat_file)
        object_image_feats = _load_features(self.object_image_feat_file)

        # Pixel area features: log container px, log mean object px, log ratio
        scene = self.samples[idx]
        frame_id_path = scene / "sam_data" / "frame_id.txt"
        inst_px_path = scene / "sam_data" / "instance_px.pt"

        # Use obj_seg pile mask (filled interior) for container pixel area
        cont_px = 0.0
        if frame_id_path.exists():
            fid = frame_id_path.read_text().strip()
            pile_mask_path = scene / "obj_seg" / f"Objects_Mask{fid}.png"
            if pile_mask_path.exists():
                cont_px = float(
                    (np.array(Image.open(pile_mask_path).convert("L")) > 0).sum()
                )

        obj_px = 0.0
        if inst_px_path.exists():
            try:
                inst_px = torch.load(
                    inst_px_path, map_location="cpu", weights_only=True
                )
                obj_px = float(inst_px.float().mean().item())
            except Exception:
                pass

        pixel_feats = _compute_pixel_feats(cont_px, obj_px)

        # Packing factor from simulation ground truth (volume_ratio_no_edges)
        packing_factor = None
        sim_path = scene / "simulation_results.json"
        if sim_path.exists():
            with open(sim_path) as f:
                sim = json.load(f)
            pf = sim.get("volume_ratio_no_edges")
            if pf is not None:
                packing_factor = torch.tensor(float(pf), dtype=torch.float32)

        return {
            "container_outputs": data["container"],
            "object_outputs": data["object"],
            "true_count": torch.tensor(data["true_count"], dtype=torch.float32),
            "geom_features": data["geom_features"],
            "geom_estimate": torch.tensor(data["geom_estimate"], dtype=torch.float32),
            "image_feats": image_feats,
            "container_image_feats": container_image_feats,
            "object_image_feats": object_image_feats,
            "pixel_feats": pixel_feats,
            "packing_factor": packing_factor,
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


def _compute_pixel_feats(cont_px: float, obj_px: float) -> torch.Tensor:
    log_cont = float(np.log(cont_px + 1))
    log_obj = float(np.log(obj_px + 1))
    log_ratio = log_cont - log_obj
    return torch.tensor([log_cont, log_obj, log_ratio], dtype=torch.float32)


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
        "pixel_feats": torch.stack([b["pixel_feats"] for b in batch]),
        "packing_factor": (
            torch.stack([b["packing_factor"] for b in batch])
            if all(b["packing_factor"] is not None for b in batch)
            else None
        ),
        "sample_name": [b["sample_name"] for b in batch],
    }
