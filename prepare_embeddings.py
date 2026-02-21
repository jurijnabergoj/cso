import os
os.environ["LIDRA_SKIP_INIT"] = "true"

from sam3d_objects.pipeline.inference_with_embeddings import InferenceWithEmbeddings
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
import torch
import json
import time


def prepare_data(data_dir: set, pipeline: InferenceWithEmbeddings):
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
        
        save_dict = {
            "container": container_out,
            "object": object_out,
            "true_count": gt_count
        }
        
        torch.save(save_dict, scene / "embeddings.pt")
        print(f"Saved embeddings for {scene.name}")
        
    return
    
    
if __name__ == "__main__":
    PATH = os.getcwd()
    TAG = "hf"
    
    config_path = f"{PATH}/sam-3d-objects/checkpoints/{TAG}/pipeline.yaml"
    pipeline = InferenceWithEmbeddings(config_path, compile=False)

    data_dir = Path("../../../projects/FRI/jn16867/3d-counting/scenes_part1")
    
    print(f"Preparing data")
    start_time = time.time()
    
    prepare_data(data_dir=data_dir, pipeline=pipeline)
    
    end_time = time.time() - start_time
    print(f"Finished preparing data, time taken: {end_time} seconds")
    