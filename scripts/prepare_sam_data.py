"""
Prepare per-scene data needed for SAM-based instance feature extraction.

For each scene, selects the best frame (maximum visible object pixels)
and writes sam_data/frame_id.txt.

generate_instance_features.py reads this file and loads images/masks
directly from the original scene directories — no data is duplicated.

No GPU required. Run before generate_instance_features.py.

Usage:
    python scripts/prepare_sam_data.py --data-dir /path/to/scenes
"""

import os
import time
import argparse
import numpy as np
from pathlib import Path
from PIL import Image


def get_best_frame(image_dir, obj_seg_dir, floor_seg_dir, box_seg_dir):
    """Return the frame id with the most visible object pixels.

    All four corresponding files must exist:
      images/RGB{id}.jpg
      obj_seg/Objects_Mask{id}.png
      floor_seg/Ground_Mask{id}.png
      box_seg/Box_Mask{id}.png
    """
    floor_names = {f.name for f in Path(floor_seg_dir).iterdir()}
    box_names = {f.name for f in Path(box_seg_dir).iterdir()}

    best_id = None
    best_score = -1

    for obj_file in sorted(Path(obj_seg_dir).iterdir()):
        fid = obj_file.stem[-4:]  # e.g. "0010" from Objects_Mask0010.png
        if f"Ground_Mask{fid}.png" not in floor_names:
            continue
        if f"Box_Mask{fid}.png" not in box_names:
            continue
        if not (Path(image_dir) / f"RGB{fid}.jpg").exists():
            continue

        score = int((np.array(Image.open(obj_file).convert("L")) > 0).sum())
        if score > best_score:
            best_score = score
            best_id = fid

    if best_id is None:
        raise RuntimeError(f"No valid frame found in {obj_seg_dir}")

    return best_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-process even if sam_data/frame_id.txt already exists",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    n_done = n_skip = n_existing = 0
    t0 = time.time()

    for category in sorted(os.listdir(data_dir)):
        scene_dir = data_dir / category
        sam_dir = scene_dir / "sam_data"
        frame_id_path = sam_dir / "frame_id.txt"

        if not args.overwrite and frame_id_path.exists():
            print(f"  skip {category} (already done)")
            n_existing += 1
            continue

        image_dir = scene_dir / "images"
        obj_seg_dir = scene_dir / "obj_seg"
        floor_seg_dir = scene_dir / "floor_seg"
        box_seg_dir = scene_dir / "box_seg"

        required = {
            "images": image_dir,
            "obj_seg": obj_seg_dir,
            "floor_seg": floor_seg_dir,
            "box_seg": box_seg_dir,
        }
        missing = [name for name, d in required.items() if not d.exists()]
        if missing:
            print(f"  skip {category}: missing {missing}")
            n_skip += 1
            continue

        try:
            fid = get_best_frame(image_dir, obj_seg_dir, floor_seg_dir, box_seg_dir)
        except RuntimeError as e:
            print(f"  skip {category}: {e}")
            n_skip += 1
            continue

        print(f"  {category}  frame={fid}")
        sam_dir.mkdir(parents=True, exist_ok=True)
        frame_id_path.write_text(fid)
        n_done += 1

    elapsed = time.time() - t0
    print(
        f"\nDone: {n_done} processed, {n_existing} skipped (existing), "
        f"{n_skip} skipped (missing data), {elapsed:.0f}s"
    )
