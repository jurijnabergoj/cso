import os
os.environ["LIDRA_SKIP_INIT"] = "true"

from cso.embeddings.inference_with_embeddings import InferenceWithEmbeddings, extract_geometric_features, compute_geometric_count_estimate
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import torch
import json
import time
import argparse


def generate_embeddings(data_dir: set, pipeline: InferenceWithEmbeddings):
    job_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num_jobs = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
    
    print(f"{job_id=}")
    print(f"{num_jobs=}")

    scenes = sorted(data_dir.iterdir())
    scenes = scenes[job_id::num_jobs]

    for scene in sorted(data_dir.iterdir()):
        if not os.path.exists(scene / "geco2_data"):
            print(f"Skipping scene {scene}, no geco2_data found")
            continue
        
        if os.path.exists(scene / "embeddings.pt"):
            print(f"Skipping scene {scene}, embeddings already computed")
            continue
        
        image = np.array(Image.open(scene / "geco2_data" / "image.png").convert("RGBA"))
        obj_mask = np.array(Image.open(scene / "geco2_data" / "obj_mask.png").convert("L"))
        box_mask = np.array(Image.open(scene / "geco2_data" / "box_mask.png").convert("L"))
        
        obj_mask = (obj_mask > 0).astype(np.uint8) * 255
        box_mask = (box_mask > 0).astype(np.uint8) * 255
        
        object_out = pipeline.run_with_embeddings(image, obj_mask, seed=42)
        container_out = pipeline.run_with_embeddings(image, box_mask, seed=42)
        
        with open(scene / "gt_count.json") as f:
            gt_count = json.load(f)
        
        embeddings = {
            "container": container_out,
            "object": object_out,
        }
        
        embeddings["geom_features"] = extract_geometric_features(embeddings["container"], embeddings["object"])
        embeddings["geom_estimate"] = compute_geometric_count_estimate(embeddings["container"], embeddings["object"])
        embeddings["true_count"] = gt_count
        
        torch.save(embeddings, scene / "embeddings.pt")
        print(f"Saved embeddings for {scene.name}")
        
    return
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--tag", default="hf")
    args = parser.parse_args()
    
    PROJECT_ROOT = Path(__file__).resolve().parents[1]  # cso/
    config_path = PROJECT_ROOT / "ext" / "sam-3d-objects" / "checkpoints" / args.tag / "pipeline.yaml"

    pipeline = InferenceWithEmbeddings(config_path, compile=False)
    
    train_data_dir = Path("/d/hpc/projects/FRI/jn16867/3d-counting/scenes_part1")
    val_data_dir = Path("/d/hpc/projects/FRI/jn16867/3d-counting/scenes_val")
    
    generate_embeddings(data_dir=args.data_dir, pipeline=pipeline)
    print(f"Finished generating embeddings.")
    