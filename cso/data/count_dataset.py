from torch.utils.data import Dataset
from PIL import Image
import json
import math
import numpy as np
import torch
from pathlib import Path

_SLAT_SEQ_K = 256  # number of voxels to subsample per object for cross-attention


class CountDataset(Dataset):
    def __init__(
        self,
        data_dirs,
        image_feat_file="image_feats.pt",
        container_image_feat_file="",
        object_image_feat_file="",
        use_clean_geom=False,
    ):
        if isinstance(data_dirs, (str, Path)):
            data_dirs = [data_dirs]

        self.data_dirs = [Path(d) for d in data_dirs]
        self.image_feat_file = image_feat_file
        self.container_image_feat_file = container_image_feat_file
        self.object_image_feat_file = object_image_feat_file
        self.use_clean_geom = use_clean_geom

        self.samples = []
        for dir in self.data_dirs:
            self.samples.extend(
                [
                    d
                    for d in dir.iterdir()
                    if (d / "sam3d_data" / "embeddings.pt").exists()
                    and (d / "sam_data").exists()
                ]
            )
        self.samples = sorted(self.samples)

    def __len__(self):
        return len(self.samples)

    def _load(self, idx):
        scene = self.samples[idx]

        def _load_features(filename):
            if not filename:
                return None
            p = scene / filename
            if not p.exists():
                return None
            feat = torch.load(p, map_location="cpu", weights_only=True)
            # Mean-pool instance features (k, D) -> (D,)
            if feat.dim() == 2:
                feat = feat.float().mean(dim=0)
            return feat

        # Image features
        image_feats_path = scene / self.image_feat_file
        image_feats = (
            torch.load(image_feats_path, map_location="cpu", weights_only=True)
            if image_feats_path.exists()
            else None
        )
        container_image_feats = _load_features(self.container_image_feat_file)
        object_image_feats = _load_features(self.object_image_feat_file)

        # Pixel area features: log container px, log mean object px, log ratio
        frame_id_path = scene / "sam_data" / "frame_id.txt"
        inst_px_path = scene / "sam_data" / "instance_px.pt"

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
                inst_px = torch.load(inst_px_path, map_location="cpu", weights_only=True)
                obj_px = float(inst_px.float().mean().item())
            except Exception:
                pass  # corrupted file — fall back to obj_px=0

        pixel_feats = _compute_pixel_feats(cont_px, obj_px)

        # Packing factor from simulation ground truth
        packing_factor = None
        sim_path = scene / "simulation_results.json"
        if sim_path.exists():
            with open(sim_path) as f:
                sim = json.load(f)
            pf = sim.get("volume_ratio_no_edges")
            if pf is not None:
                packing_factor = torch.tensor(float(pf), dtype=torch.float32)

        # SAM3D embeddings
        sam3d_path = scene / "sam3d_data" / "embeddings.pt"
        try:
            sam3d_data = torch.load(sam3d_path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Error loading sam3d embeddings at {sam3d_path} (idx={idx})")
            raise e

        if self.use_clean_geom:
            geom_features = _compute_clean_geom_features(
                sam3d_data["container"], sam3d_data["object"]
            )
        else:
            geom_features = sam3d_data["geom_features"]

        return {
            "container_outputs": sam3d_data["container"],
            "object_outputs": sam3d_data["object"],
            "true_count": torch.tensor(sam3d_data["true_count"], dtype=torch.float32),
            "geom_features": geom_features,
            "geom_estimate": torch.tensor(sam3d_data["geom_estimate"], dtype=torch.float32),
            "image_feats": image_feats,
            "container_image_feats": container_image_feats,
            "object_image_feats": object_image_feats,
            "pixel_feats": pixel_feats,
            "packing_factor": packing_factor,
            "sample_name": scene.name,
        }

    def __getitem__(self, idx):
        return self._load(idx)


# ── helpers ──────────────────────────────────────────────────────────────────

def _pool_slat(x):
    """Max + mean pool (N, D) → (2*D,)."""
    return torch.cat([x.max(dim=0)[0], x.mean(dim=0)], dim=-1)


def _subsample_slat(x, K=_SLAT_SEQ_K):
    """Randomly subsample K rows from (N, D) → (K, D). Oversamples if N < K."""
    N = x.shape[0]
    if N >= K:
        idx = torch.randperm(N)[:K]
    else:
        idx = torch.randint(0, N, (K,))
    return x[idx]


def _compute_pixel_feats(cont_px: float, obj_px: float) -> torch.Tensor:
    log_cont = float(np.log(cont_px + 1))
    log_obj = float(np.log(obj_px + 1))
    log_ratio = log_cont - log_obj
    return torch.tensor([log_cont, log_obj, log_ratio], dtype=torch.float32)


def _compute_clean_geom_features(container: dict, obj: dict) -> torch.Tensor:
    """6 clean, non-redundant geometric features derived from SAM3D coords and scale.

    Features:
      0  log(N_cont_vox + 1)           — container voxel count (proxy for volume)
      1  log(N_obj_vox + 1)            — object voxel count
      2  log(N_cont_vox / N_obj_vox + 1) — voxel count ratio (r=0.42 with true count)
      3  container bbox aspect ratio   — max/min of bbox dimensions
      4  object bbox aspect ratio
      5  log(cont_scale / obj_scale)   — 1D scale ratio (avoids cube amplification)
    """
    def aspect_ratio(coords):
        xyz = coords[:, 1:].float()
        bbox = xyz.max(0).values - xyz.min(0).values + 1
        return float(bbox.max() / (bbox.min() + 1e-6))

    N_cont = container["coords"].shape[0]
    N_obj = obj["coords"].shape[0]
    cont_scale = float(container["scale"][0])
    obj_scale = float(obj["scale"][0])

    return torch.tensor([
        math.log(N_cont + 1),
        math.log(N_obj + 1),
        math.log(N_cont / max(N_obj, 1) + 1),
        aspect_ratio(container["coords"]),
        aspect_ratio(obj["coords"]),
        math.log(max(cont_scale / max(obj_scale, 1e-6), 1e-6)),
    ], dtype=torch.float32)


def collate_fn(batch):
    return {
        "container_outputs": {
            "coords": [b["container_outputs"]["coords"] for b in batch],
            "slat_features": torch.stack(
                [_pool_slat(b["container_outputs"]["slat_features"]) for b in batch]
            ),
            "slat_seq": torch.stack(
                [_subsample_slat(b["container_outputs"]["slat_features"]) for b in batch]
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
            "slat_seq": torch.stack(
                [_subsample_slat(b["object_outputs"]["slat_features"]) for b in batch]
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
