from pathlib import Path
from sam3d_objects.pipeline.inference_with_embeddings import InferenceWithEmbeddings
from PIL import Image
import numpy as np
import torch
import json


def preprocess_sample(data_dir: Path, pipeline: InferenceWithEmbeddings, frame: str):
    image = np.array(Image.open(data_dir / "images" / frame + ".jpg").convert("RGB"))
    
    container_mask = np.array(Image.open(data_dir / "box_seg" / first_frame).convert("L"))
    container_mask = (container_mask > 0).astype(np.uint8) * 255
    
    object_mask = np.array(Image.open(data_dir / "geco2_mask" / "mask.png"))
    object_mask = (object_mask > 0).astype(np.uint8) * 255
    
    with open(data_dir / "info.json") as f:
        data = json.load(f)
        true_count = data['gt_count']
    
    print(f"Computing embeddings for container")
    container_out = pipeline.run_with_embeddings(image, container_mask, seed=42)
    print(f"Computing embeddings for object")
    object_out = pipeline.run_with_embeddings(image, object_mask, seed=42)
    
    save_dict = {
        "container": container_out,
        "object": object_out,
        "true_count": true_count
    }
    
    torch.save(save_dict, data_dir / "embeddings.pt")
    print(f"Saved embeddings for {data_dir.name}")
    

def preprocess_sample_test(data_dir: Path, pipeline: InferenceWithEmbeddings):
    image = np.array(Image.open(data_dir / "geco2_mask" / "image.png").convert("RGB"))
    first_frame = "frame_00001.png"
    
    if (data_dir / "box_seg").exists():
        container_mask = np.array(Image.open(data_dir / "box_seg" / first_frame).convert("L"))
    elif (data_dir / "obj_seg").exists():
        container_mask = np.array(Image.open(data_dir / "obj_seg" / first_frame).convert("L"))
    else:
        print(f"No box or object segmentation found for data_dir: {data_dir}. Skipping category.")
        return
    container_mask = (container_mask > 0).astype(np.uint8) * 255
    
    object_mask = np.array(Image.open(data_dir / "geco2_mask" / "mask.png"))
    object_mask = (object_mask > 0).astype(np.uint8) * 255
    
    with open(data_dir / "info.json") as f:
        data = json.load(f)
        true_count = data['gt_count']
    
    print(f"Computing embeddings for container")
    container_out = pipeline.run_with_embeddings(image, container_mask, seed=42)
    print(f"Computing embeddings for object")
    object_out = pipeline.run_with_embeddings(image, object_mask, seed=42)
    
    save_dict = {
        "container": container_out,
        "object": object_out,
        "true_count": true_count
    }
    
    torch.save(save_dict, data_dir / "embeddings.pt")
    print(f"Saved embeddings for {data_dir.name}")
    
    def invert_mask(mask) {
        inverted_mask = mask.copy()
        inverted_mask
    }