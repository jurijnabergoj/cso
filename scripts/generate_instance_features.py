"""
Generate per-scene DINOv2-S features from the best camera frame (sam_data/frame_id.txt).

All outputs use the same frame so features are spatially consistent.

Pipeline per scene:
  1. Read sam_data/frame_id.txt → load images/RGB{fid}.jpg
  2. Scene-level DINOv2-S CLS token → sam_data/dinov2_s_feats.pt          (384,)
  3. Container masked crop (box_seg/Box_Mask{fid}.png) → sam_data/dinov2_s_container_feats.pt  (384,)
  4. Distance-transform pile mask → N peak points (most isolated object centres)
  5. For each point: SAM2 point-prompt → instance mask
  6. Crop image to instance bbox, blacken non-instance pixels → DINOv2-S CLS token
     → sam_data/dinov2_s_instance_feats.pt  (k, 384)
     → sam_data/instance_px.pt              (k,)    — pixel area per instance

Requires GPU. Run prepare_sam_data.py first.

Usage:
    python scripts/generate_instance_features.py \\
        --data-dir /path/to/scenes \\
        --sam-checkpoint /path/to/sam2_hiera_base_plus.pt \\
        [--sam-config sam2_hiera_b+.yaml] [--n-instances 5] [--min-distance 15] [--overwrite]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max


BACKBONE_INFO = {
    "dinov2_s": ("facebook/dinov2-small", 384),
    "dinov2_b": ("facebook/dinov2-base", 768),
}


def build_dinov2(backbone: str, device):
    from transformers import AutoModel

    hf_name, _ = BACKBONE_INFO[backbone]
    model = AutoModel.from_pretrained(hf_name)
    model.eval()
    return model.to(device)


# Maps checkpoint stem → SAM2 hydra config name
_CKPT_TO_CONFIG = {
    "sam2.1_hiera_tiny": "sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small": "sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus": "sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large": "sam2.1/sam2.1_hiera_l.yaml",
    "sam2_hiera_tiny": "sam2_hiera_t.yaml",
    "sam2_hiera_small": "sam2_hiera_s.yaml",
    "sam2_hiera_base_plus": "sam2_hiera_b+.yaml",
    "sam2_hiera_large": "sam2_hiera_l.yaml",
}


def build_sam2_predictor(checkpoint: Path, config: str | None, device):
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from sam2.sam2.build_sam import build_sam2
    from sam2.sam2.sam2_image_predictor import SAM2ImagePredictor
    import sam2.sam2 as _sam2_pkg

    if config is None:
        config = _CKPT_TO_CONFIG.get(checkpoint.stem)
        if config is None:
            raise ValueError(
                f"Cannot infer SAM2 config from '{checkpoint.stem}'. "
                "Pass --sam-config explicitly."
            )

    # build_sam2 calls hydra.compose; initialize GlobalHydra pointing at sam2_configs/
    sam2_configs_dir = str(
        Path(_sam2_pkg.__file__).resolve().parent.parent / "sam2_configs"
    )
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=sam2_configs_dir, job_name="sam2")

    model = build_sam2(config, ckpt_path=str(checkpoint), device=device)
    return SAM2ImagePredictor(model)


_DINO_TRANSFORM = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@torch.no_grad()
def _encode(dino_model, pil_img: Image.Image, device) -> torch.Tensor:
    x = _DINO_TRANSFORM(pil_img).unsqueeze(0).to(device)
    feat = dino_model(x).last_hidden_state[:, 0, :].squeeze()  # CLS (384,)
    return feat.float().cpu()


def crop_and_encode(
    dino_model,
    img_rgb: np.ndarray,
    mask: np.ndarray,
    device,
    pad: int = 10,
) -> torch.Tensor:
    """Crop to mask bbox, blacken non-mask pixels, encode with DINOv2."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return _encode(dino_model, Image.fromarray(img_rgb), device)
    H, W = img_rgb.shape[:2]
    y1 = max(0, int(rows.min()) - pad)
    y2 = min(H, int(rows.max()) + pad + 1)
    x1 = max(0, int(cols.min()) - pad)
    x2 = min(W, int(cols.max()) + pad + 1)
    crop = img_rgb[y1:y2, x1:x2].copy()
    crop[~mask[y1:y2, x1:x2]] = 0
    return _encode(dino_model, Image.fromarray(crop), device)


def find_point_prompts(pile_mask: np.ndarray, n: int, min_distance: int) -> np.ndarray:
    """
    Distance-transform peaks on pile_mask.
    Returns (k, 2) int array of (row, col) coordinates, k <= n.
    """
    dt = distance_transform_edt(pile_mask > 0)
    peaks = peak_local_max(dt, min_distance=min_distance, num_peaks=n)
    return peaks  # (k, 2)


def get_instance_mask(
    predictor, row: int, col: int, pile_mask: np.ndarray
) -> np.ndarray:
    """
    SAM2 single-point prompt at (row, col). Returns bool mask (H, W).
    Among the three candidate masks, picks the highest-scoring one that
    overlaps the pile region.
    """
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[col, row]]),  # SAM takes (x=col, y=row)
        point_labels=np.array([1]),
        multimask_output=True,
    )
    pile_bool = pile_mask > 0
    for idx in np.argsort(scores)[::-1]:
        if np.logical_and(masks[idx], pile_bool).sum() > 0:
            return masks[idx].astype(bool)
    return masks[np.argmax(scores)].astype(bool)


def process_scene(
    scene_dir: Path,
    predictor,
    dino_model,
    device,
    n_instances: int,
    min_distance: int,
    overwrite: bool,
    backbone: str = "dinov2_s",
) -> str:
    sam_dir = scene_dir / "sam_data"
    out_feats = sam_dir / f"{backbone}_instance_feats.pt"
    out_px = sam_dir / "instance_px.pt"
    out_scene = sam_dir / f"{backbone}_feats.pt"
    out_container = sam_dir / f"{backbone}_container_feats.pt"
    out_best_view_image = sam_dir / "best_instance_mask.png"

    all_done = (
        out_feats.exists()
        and out_px.exists()
        and out_scene.exists()
        and out_container.exists()
        and out_best_view_image.exists()
    )
    if all_done and not overwrite:
        return "skip"

    frame_id_path = sam_dir / "frame_id.txt"
    if not frame_id_path.exists():
        return "missing_frame_id"

    fid = frame_id_path.read_text().strip()
    img_path = scene_dir / "images" / f"RGB{fid}.jpg"
    mask_path = scene_dir / "obj_seg" / f"Objects_Mask{fid}.png"
    box_mask_path = scene_dir / "box_seg" / f"Box_Mask{fid}.png"

    if not img_path.exists() or not mask_path.exists():
        return "missing_input"

    img_rgb = np.array(Image.open(img_path).convert("RGB"))
    pile_mask = np.array(Image.open(mask_path).convert("L"))

    # Scene-level DINOv2 feature (CLS token of full image)
    if not out_scene.exists() or overwrite:
        scene_feat = _encode(dino_model, Image.fromarray(img_rgb), device)
        torch.save(scene_feat, out_scene)

    # Container DINOv2 feature (masked crop from box_seg)
    if (not out_container.exists() or overwrite) and box_mask_path.exists():
        box_mask = np.array(Image.open(box_mask_path).convert("L")) > 0
        container_feat = crop_and_encode(dino_model, img_rgb, box_mask, device)
        torch.save(container_feat, out_container)

    if (pile_mask > 0).sum() < 16:
        return "empty_pile_mask"

    if out_feats.exists() and out_px.exists() and out_best_view_image.exists() and not overwrite:
        return "ok (image feats only)"

    points = find_point_prompts(pile_mask, n=n_instances, min_distance=min_distance)
    if len(points) == 0:
        return "no_points"

    feats, areas, inst_masks = [], [], []
    with torch.inference_mode():
        predictor.set_image(img_rgb)
        for row, col in points:
            inst_mask = get_instance_mask(predictor, int(row), int(col), pile_mask)
            area = int(inst_mask.sum())
            if area < 16:  # skip tiny spurious blobs
                continue
            feat = crop_and_encode(dino_model, img_rgb, inst_mask, device)
            feats.append(feat)
            areas.append(area)
            inst_masks.append(inst_mask)

    if len(feats) == 0:
        return "no_valid_instances"

    # Save best instance mask + pixel areas — backbone-independent, skip if already exist
    if not out_best_view_image.exists() or overwrite:
        best_idx = int(np.argmax(areas))
        best_mask_img = Image.fromarray(inst_masks[best_idx].astype(np.uint8) * 255)
        best_mask_img.save(out_best_view_image)

    if not out_px.exists() or overwrite:
        torch.save(torch.tensor(areas, dtype=torch.float32), out_px)

    torch.save(torch.stack(feats), out_feats)
    return f"ok ({len(feats)}/{len(points)} instances)"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--backbone",
        default="dinov2_s",
        choices=list(BACKBONE_INFO.keys()),
        help="DINOv2 backbone variant (default: dinov2_s)",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        required=True,
        help="Path to SAM checkpoint .pth file",
    )
    parser.add_argument(
        "--sam-config",
        default=None,
        help="SAM2 hydra config name (e.g. 'sam2.1/sam2.1_hiera_b+.yaml'). "
        "Auto-detected from checkpoint name if omitted.",
    )
    parser.add_argument(
        "--n-instances",
        type=int,
        default=5,
        help="Number of SAM point prompts per scene (default: 5)",
    )
    parser.add_argument(
        "--min-distance",
        type=int,
        default=15,
        help="Min pixel distance between distance-transform peaks (default: 15)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading {args.backbone} ...")
    dino_model = build_dinov2(args.backbone, device)

    print(f"Loading SAM2 from {args.sam_checkpoint} ...")
    predictor = build_sam2_predictor(args.sam_checkpoint, args.sam_config, device)

    scenes = sorted(s for s in args.data_dir.iterdir() if s.is_dir())
    print(f"Processing {len(scenes)} scenes")

    t0 = time.time()
    n_ok = n_skip = n_err = 0

    for scene in scenes:
        try:
            status = process_scene(
                scene,
                predictor,
                dino_model,
                device,
                n_instances=args.n_instances,
                min_distance=args.min_distance,
                overwrite=args.overwrite,
                backbone=args.backbone,
            )
        except Exception as e:
            n_err += 1
            print(f"  {scene.name}: ERROR — {e}")
            continue
        if status == "skip":
            n_skip += 1
            print(f"  skip {scene.name}")
        elif status.startswith("ok"):
            n_ok += 1
            print(f"  {scene.name}: {status}")
        else:
            n_err += 1
            print(f"  {scene.name}: {status}")

    elapsed = time.time() - t0
    print(f"\nDone: {n_ok} ok, {n_skip} skipped, {n_err} errors  ({elapsed:.0f}s)")
