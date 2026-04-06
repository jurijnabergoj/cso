"""
Generate SAM3D shape embeddings for each scene using the best camera frame.

Uses the same best frame as the SAM2 instance pipeline (sam_data/frame_id.txt)
so all embeddings are spatially consistent.

Masks:
  container → box_seg/Box_Mask{fid}.png   (actual container walls, not pile)
  object    → sam_data/best_instance_mask.png  (largest SAM2 instance)

Output: sam3d_data/embeddings.pt  (same structure as legacy embeddings.pt)

Requires: prepare_sam_data.py and generate_instance_features.py must have run first
          (generate_instance_features.py writes sam_data/best_instance_mask.png).

Usage:
    python scripts/generate_shape_embeddings.py --data-dir /path/to/scenes [--overwrite]
"""

import os

os.environ["LIDRA_SKIP_INIT"] = "true"

from cso.embeddings.inference_with_embeddings import (
    InferenceWithEmbeddings,
    extract_geometric_features,
)
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import json
import time
import argparse


def is_mask_empty(mask: np.ndarray):
    return np.count_nonzero(mask) == 0


def generate_embeddings(
    data_dir: Path,
    pipeline: InferenceWithEmbeddings,
    overwrite: bool,
    shard_id: int = 0,
    num_shards: int = 1,
):
    scenes = sorted(data_dir.iterdir())
    scenes = scenes[shard_id::num_shards]

    n_done = n_skip = n_err = 0
    t0 = time.time()

    for scene in scenes:
        if not scene.is_dir():
            continue

        sam3d_dir = scene / "sam3d_data"
        out_path = sam3d_dir / "embeddings.pt"

        if out_path.exists() and not overwrite:
            print(f"  skip {scene.name} (already done)")
            n_skip += 1
            continue

        frame_id_path = scene / "sam_data" / "frame_id.txt"
        if not frame_id_path.exists():
            print(
                f"  skip {scene.name}: no sam_data/frame_id.txt (run prepare_sam_data.py first)"
            )
            n_err += 1
            continue

        fid = frame_id_path.read_text().strip()
        img_path = scene / "images" / f"RGB{fid}.jpg"
        container_mask_path = scene / "box_seg" / f"Box_Mask{fid}.png"
        object_mask_path = scene / "sam_data" / "best_instance_mask.png"
        gt_count_path = scene / "gt_count.json"

        missing = [
            p
            for p in [img_path, container_mask_path, object_mask_path, gt_count_path]
            if not p.exists()
        ]
        if missing:
            names = [p.name for p in missing]
            print(
                f"  skip {scene.name}: missing {names} (run generate_instance_features.py first)"
            )
            n_err += 1
            continue

        image = np.array(Image.open(img_path).convert("RGBA"))
        container_mask = (
            np.array(Image.open(container_mask_path).convert("L")) > 0
        ).astype(np.uint8) * 255
        object_mask = (np.array(Image.open(object_mask_path).convert("L")) > 0).astype(
            np.uint8
        ) * 255

        if is_mask_empty(container_mask):
            print(f"  skip {scene.name}: empty container mask")
            n_err += 1
            continue
        if is_mask_empty(object_mask):
            print(f"  skip {scene.name}: empty object mask")
            n_err += 1
            continue

        try:
            object_out = pipeline.run_with_embeddings(image, object_mask, seed=42)
            container_out = pipeline.run_with_embeddings(image, container_mask, seed=42)
        except Exception as e:
            print(f"  error {scene.name}: SAM3D failed — {e}")
            n_err += 1
            continue

        with open(gt_count_path) as f:
            gt_count = json.load(f)

        # Scale-based count estimate: 0.64 × (container_vol / object_vol)
        # Uses real-world scale in metres; more reliable than voxel-coord ratio.
        cont_scale = container_out["scale"]  # (3,) metres
        obj_scale = object_out["scale"]      # (3,) metres
        cont_vol = float(torch.as_tensor(cont_scale).prod())
        obj_vol = float(torch.as_tensor(obj_scale).prod())
        scale_geom_estimate = 0.64 * cont_vol / max(obj_vol, 1e-6)

        embeddings = {
            "container": container_out,
            "object": object_out,
            "geom_features": extract_geometric_features(container_out, object_out),
            "geom_estimate": scale_geom_estimate,
            "true_count": gt_count,
        }

        sam3d_dir.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, out_path)
        print(f"  saved {scene.name}")
        n_done += 1

    elapsed = time.time() - t0
    print(f"\nDone: {n_done} saved, {n_skip} skipped, {n_err} errors  ({elapsed:.0f}s)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tag", default="hf")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    config_path = (
        PROJECT_ROOT
        / "ext"
        / "sam-3d-objects"
        / "checkpoints"
        / args.tag
        / "pipeline.yaml"
    )

    pipeline = InferenceWithEmbeddings(config_path, compile=False)

    generate_embeddings(
        data_dir=args.data_dir,
        pipeline=pipeline,
        overwrite=args.overwrite,
        shard_id=args.shard_id,
        num_shards=args.num_shards,
    )

    print("Finished generating embeddings.")
