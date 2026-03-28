import os
import argparse
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

BACKBONE_INFO = {
    # (feat_dim, output_filename)
    "resnet50": (2048, "image_feats.pt"),
    "dinov2_s": (384, "dinov2_s_feats.pt"),
    "dinov2_b": (768, "dinov2_b_feats.pt"),
}

"""
Pre-compute image features for all scenes using a frozen pretrained backbone.

Scene-level mode (default):
  Encodes geco2_data/image.png → saved as <backbone>_feats.pt

Masked-crop mode (--masked):
  Crops the scene image to each object's bounding box using geco2_data masks,
  then encodes container and object separately:
    geco2_data/box_mask.npy + image.png → <backbone>_container_feats.pt
    geco2_data/obj_mask.npy  + image.png → <backbone>_object_feats.pt

Supported backbones:
  resnet50 -> ResNet50 avgpool, 2048-dim
  dinov2_s -> DINOv2 ViT-S/14 CLS token, 384-dim
  dinov2_b -> DINOv2 ViT-B/14 CLS token, 768-dim

Usage:
    python scripts/generate_image_embeddings.py --data-dir /path/to/scenes --backbone dinov2_s --masked
"""


def build_resnet50(device):
    from torchvision.models import resnet50, ResNet50_Weights

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    # Remove final FC; keep AdaptiveAvgPool2d
    encoder = torch.nn.Sequential(*list(model.children())[:-1])
    encoder.eval()
    return encoder.to(device)


def build_dinov2(variant: str, device):
    from transformers import AutoModel

    hf_name = {"dinov2_s": "facebook/dinov2-small", "dinov2_b": "facebook/dinov2-base"}[
        variant
    ]
    model = AutoModel.from_pretrained(hf_name)
    model.eval()
    return model.to(device)


def get_transform(backbone: str):
    if backbone == "resnet50":
        return T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        # backbone == "dinov2_":
        return T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


@torch.no_grad()
def _encode(
    encoder, transform, pil_img: Image.Image, device, backbone: str
) -> torch.Tensor:
    x = transform(pil_img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
    if backbone == "resnet50":
        feat = encoder(x).squeeze()  # (2048,)
    else:
        out = encoder(x)
        feat = out.last_hidden_state[:, 0, :].squeeze()  # (384,) or (768,)
    return feat.float().cpu()


def extract_feat(
    encoder, transform, img_path: Path, device, backbone: str
) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    return _encode(encoder, transform, img, device, backbone)


def extract_masked_feat(
    encoder,
    transform,
    img_path: Path,
    mask_npy_path: Path,
    device,
    backbone: str,
    crop_pad: int = 10,
) -> torch.Tensor:
    """
    Crop the scene image to the object's bounding box using a binary mask,
    then extract features from the masked crop.
    """
    img_arr = np.array(Image.open(img_path).convert("RGB"))
    mask = np.load(mask_npy_path).astype(bool)

    if mask.ndim == 3:
        mask = mask[..., 0]

    rows, cols = np.where(mask)
    if len(rows) == 0:
        # mask is empty so use full image
        pil_img = Image.fromarray(img_arr)
    else:
        H, W = img_arr.shape[:2]
        y1 = max(0, int(rows.min()) - crop_pad)
        y2 = min(H, int(rows.max()) + crop_pad + 1)
        x1 = max(0, int(cols.min()) - crop_pad)
        x2 = min(W, int(cols.max()) + crop_pad + 1)

        cropped = img_arr[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2]
        cropped[~crop_mask] = 0
        pil_img = Image.fromarray(cropped)

    return _encode(encoder, transform, pil_img, device, backbone)


def run(
    data_dir: Path,
    encoder,
    transform,
    device,
    backbone: str,
    overwrite: bool,
    masked: bool,
):
    _, out_filename = BACKBONE_INFO[backbone]
    job_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    num_jobs = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))

    scenes = sorted(data_dir.iterdir())
    scenes = scenes[job_id::num_jobs]
    print(f"Backbone: {backbone}  |  masked: {masked}")
    print(f"Processing {len(scenes)} scenes (shard {job_id}/{num_jobs})")

    if not masked:
        print(f"Output file: {out_filename}")
        _run_whole_scene(
            scenes, encoder, transform, device, backbone, out_filename, overwrite
        )
    else:
        basename = out_filename.replace("_feats.pt", "")
        cont_file = f"{basename}_container_feats.pt"
        obj_file = f"{basename}_object_feats.pt"
        print(f"Output files: {cont_file}, {obj_file}")
        _run_masked_scene(
            scenes, encoder, transform, device, backbone, cont_file, obj_file, overwrite
        )


def _run_whole_scene(
    scenes, encoder, transform, device, backbone, out_filename, overwrite
):
    for scene in scenes:
        out_path = scene / out_filename
        if out_path.exists() and not overwrite:
            existing = torch.load(out_path, map_location="cpu", weights_only=True)
            if torch.isfinite(existing).all():
                print(f"  skip {scene.name} (already done, valid)")
                continue
            print(f"  recompute {scene.name} (existing features contain NaN/Inf)")

        img_path = scene / "geco2_data" / "image.png"
        if not img_path.exists():
            print(f"  skip {scene.name} (no geco2_data/image.png)")
            continue

        feat = extract_feat(encoder, transform, img_path, device, backbone)
        if not torch.isfinite(feat).all():
            print(f"  WARNING: feat for {scene.name} contains NaN/Inf — skipping save")
            continue
        torch.save(feat, out_path)
        print(f"  saved {scene.name}  feat={tuple(feat.shape)}")


def _run_masked_scene(
    scenes, encoder, transform, device, backbone, cont_file, obj_file, overwrite
):
    for scene in scenes:
        cont_path = scene / cont_file
        obj_path = scene / obj_file

        if cont_path.exists() and obj_path.exists() and not overwrite:
            cont_ok = torch.isfinite(
                torch.load(cont_path, map_location="cpu", weights_only=True)
            ).all()
            obj_ok = torch.isfinite(
                torch.load(obj_path, map_location="cpu", weights_only=True)
            ).all()
            if cont_ok and obj_ok:
                print(f"  skip {scene.name} (already done, valid)")
                continue
            print(f"  recompute {scene.name} (existing features contain NaN/Inf)")

        img_path = scene / "geco2_data" / "image.png"
        box_mask_path = scene / "geco2_data" / "box_mask.npy"
        obj_mask_path = scene / "geco2_data" / "obj_mask.npy"

        if not img_path.exists():
            print(f"  skip {scene.name} (no geco2_data/image.png)")
            continue
        if not box_mask_path.exists() or not obj_mask_path.exists():
            print(f"  skip {scene.name} (missing box_mask.npy or obj_mask.npy)")
            continue

        cont_feat = extract_masked_feat(
            encoder, transform, img_path, box_mask_path, device, backbone
        )
        obj_feat = extract_masked_feat(
            encoder, transform, img_path, obj_mask_path, device, backbone
        )

        if not torch.isfinite(cont_feat).all() or not torch.isfinite(obj_feat).all():
            print(f"  WARNING: feat for {scene.name} contains NaN/Inf — skipping save")
            continue

        torch.save(cont_feat, cont_path)
        torch.save(obj_feat, obj_path)
        print(
            f"  saved {scene.name}  cont={tuple(cont_feat.shape)}, obj={tuple(obj_feat.shape)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Root directory containing scene subdirectories",
    )
    parser.add_argument(
        "--backbone",
        default="resnet50",
        choices=list(BACKBONE_INFO.keys()),
        help="Image backbone to use (default: resnet50)",
    )
    parser.add_argument(
        "--masked",
        action="store_true",
        help="Extract masked crops per object instead of full scene image",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-compute even if output file already exists",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.backbone == "resnet50":
        encoder = build_resnet50(device)
    else:
        encoder = build_dinov2(args.backbone, device)
    transform = get_transform(args.backbone)

    run(
        args.data_dir,
        encoder,
        transform,
        device,
        args.backbone,
        overwrite=args.overwrite,
        masked=args.masked,
    )
    print("Finished.")
