import os

os.environ.pop("SSL_CERT_FILE", None)
os.environ.pop("SSL_CERT_DIR", None)

from hydra.core.global_hydra import GlobalHydra

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
    
import torch
import gradio as gr
from gradio_image_prompter import ImagePrompter
from torch.nn import DataParallel
from models.counter_infer import build_model
from utils.arg_parser import get_argparser
from utils.data import resize_and_pad
import torchvision.ops as ops
from torchvision import transforms as T
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import cv2
import time
import shutil
import argparse


# **Function to Process Image Once**
def process_image_once(inputs, enable_mask):
    model.module.return_masks = enable_mask

    image = inputs['image']
    drawn_boxes = inputs['points']
    image_tensor = torch.tensor(image).to(device)
    image_tensor = image_tensor.permute(2, 0, 1).float() / 255.0
    image_tensor = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)

    bboxes_tensor = torch.tensor([[box[0], box[1], box[2], box[3]] for box in drawn_boxes], dtype=torch.float32).to(
        device)

    img, bboxes, scale = resize_and_pad(image_tensor, bboxes_tensor, size=1024.0)
    img = img.unsqueeze(0).to(device)
    bboxes = bboxes.unsqueeze(0).to(device)

    with torch.no_grad():
        model.cuda()
        outputs, _, _, _, masks = model(img.to(device), bboxes.to(device))

    # move ALL outputs to CPU, key-by-key (handles lists/dicts safely)
    outputs = [
        {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in out.items()}
        for out in outputs
    ]

    # make sure masks is on CPU and in a consistent structure
    if enable_mask and masks is not None:
        if torch.is_tensor(masks):
            masks = masks.detach().cpu()
        elif isinstance(masks, (list, tuple)):
            masks = [m.detach().cpu() for m in masks]
    else:
        masks = None

    return image, outputs, masks, img, scale, drawn_boxes


# **Post-process and Update Output**
def post_process(image, outputs, masks, img, scale, drawn_boxes, enable_mask, threshold):
    idx = 0
    thr_inv = 1.0 / threshold  # keep your original intent

    # --- pull tensors & drop batch dim if present ---
    pred_boxes = outputs[idx]['pred_boxes']          # [1, N, 4] or [N, 4]
    box_v      = outputs[idx]['box_v']               # [1, N]    or [N]

    if pred_boxes.dim() == 3 and pred_boxes.size(0) == 1:
        pred_boxes = pred_boxes[0]                   # -> [N, 4]
    if box_v.dim() == 2 and box_v.size(0) == 1:
        box_v = box_v[0]                             # -> [N]

    # --- selection mask over N ---
    sel = box_v > (box_v.max() / thr_inv)            # [N] bool

    # handle no survivors cleanly
    if sel.sum().item() == 0:
        # just draw the user boxes and 0 count
        image_pil = Image.fromarray(image.astype(np.uint8))
        draw = ImageDraw.Draw(image_pil)
        for box in drawn_boxes:
            draw.rectangle([box[0], box[1], box[2], box[3]], outline="red", width=3)
        # counter badge
        w, h = image_pil.size
        sq = int(0.05 * w)
        x1, y1 = 10, h - sq - 10
        draw.rectangle([x1, y1, x1+sq, y1+sq], outline="black", fill="black")
        font = ImageFont.load_default()
        txt = "0"
        text_x = x1 + (sq - draw.textlength(txt, font=font)) / 2
        text_y = y1 + (sq - 10) / 2
        draw.text((text_x, text_y), txt, fill="white", font=font)
        return image_pil, 0

    # --- NMS expects [N,4] boxes and [N] scores ---
    keep = ops.nms(pred_boxes[sel], box_v[sel], 0.5)
    pred_boxes = pred_boxes[sel][keep]               # [M,4]
    box_v = box_v[sel][keep]                         # [M]

    # clamp/scale to original image coords
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    pred_boxes = (pred_boxes / scale * img.shape[-1]).tolist()

    # to PIL
    image_pil = Image.fromarray(image.astype(np.uint8))

    # --- masks (optional) ---
    if enable_mask and masks is not None:
        # get batch slice, drop batch dim if present
        base = masks[idx]
        # If base is a tensor, ensure it has 3 dims [N,H,W]
        if torch.is_tensor(base):
            if base.dim() == 4 and base.size(0) == 1:  # e.g., [1,N,H,W]
                base = base[0]  # -> [N,H,W]
        elif isinstance(base, (list, tuple)):
            # convert list of N [H,W] masks into [N,H,W] tensor
            base = torch.stack(base, dim=0)  # now base is [N,H,W]
        else:
            raise TypeError(f"Unexpected mask type: {type(base)}")


        if masks is not None:
            masks_ = base[sel][keep]
            N_masks = masks_.shape[0]
            indices = torch.randint(1, N_masks + 1, (1, N_masks), device=masks_.device).view(-1, 1, 1)
            mask_lbl = (masks_ * indices).sum(dim=0) # [H, W]
            mask_display = (
                T.Resize(
                    (int(img.shape[2] / scale), int(img.shape[3] / scale)),
                    interpolation=T.InterpolationMode.NEAREST
                )(mask_lbl.unsqueeze(0))[0]
            )[:image_pil.size[1], :image_pil.size[0]]
            
            masks_orig_shape = (
                T.Resize(
                    (int(img.shape[2] / scale), int(img.shape[3] / scale)),
                    interpolation=T.InterpolationMode.NEAREST
                )(masks_)
            )[:, :image_pil.size[1], :image_pil.size[0]]

            
            cmap = plt.cm.tab20
            norm = plt.Normalize(vmin=0, vmax=N_masks)
            rgba = cmap(norm(mask_display))
            rgba[mask_display == 0, -1] = 0
            rgba[mask_display != 0, -1] = 0.5
            overlay = Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")
            image_pil = image_pil.convert("RGBA")
            image_pil = Image.alpha_composite(image_pil, overlay)

    # --- draw boxes & user input ---
    draw = ImageDraw.Draw(image_pil)
    for box in pred_boxes:
        draw.rectangle([box[0], box[1], box[2], box[3]], outline="orange", width=2)
    # for box in drawn_boxes:
    #    draw.rectangle([box[0], box[1], box[3], box[4]], outline="red", width=3)

    # counter badge
    w, h = image_pil.size
    sq = int(0.05 * w)
    x1, y1 = 10, h - sq - 10
    draw.rectangle([x1, y1, x1+sq, y1+sq], outline="black", fill="black")
    font = ImageFont.load_default()
    txt = str(len(pred_boxes))
    text_x = x1 + (sq - draw.textlength(txt, font=font)) / 2
    text_y = y1 + (sq - 10) / 2
    draw.text((text_x, text_y), txt, fill="white", font=font)

    return image_pil, len(pred_boxes), masks_orig_shape


def initial_process(ip_data, enable_mask_val, thr):
    # ip_data is a dict from ImagePrompter: {'image': np.ndarray, 'points': [...]}
    image, outputs, masks, img, scale, drawn_boxes = process_image_once(ip_data, enable_mask_val)
    out_img, n, masks_ori = post_process(image, outputs, masks, img, scale, drawn_boxes, enable_mask_val, thr)
    return (
        out_img, n, masks_ori,  # visible outputs
        image, outputs, masks, img, scale, drawn_boxes  # states
    )
    

def get_bounding_boxes_from_mask(mask, save_dir):
    # Convert mask to uint8
    mask_uint8 = (mask * 255).astype(np.uint8)
    
    # Distance transform
    distance = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
    
    # Find peaks
    local_max = cv2.dilate(distance, np.ones((3,3)))
    local_max = (distance == local_max).astype(np.uint8)
    
    # Connected components as markers
    
    _, markers = cv2.connectedComponents(local_max)
    markers = markers.astype(np.int32)
    
    # Need a 3-channel image for watershed
    mask_bgr = cv2.cvtColor(mask_uint8, cv2.COLOR_GRAY2BGR)
    mask_ws = cv2.watershed(mask_bgr, markers)
    
    # Watershed assigns -1 to boundaries
    labels = mask_ws.copy()
    labels[labels == -1] = 0

    bboxes = []

    # Skip label 0 (background)
    for label_id in range(1, labels.max()+1):
        ys, xs = np.where(labels == label_id)
        if len(xs) == 0 or len(ys) == 0:
            continue
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        
        w, h = x_max - x_min, y_max - y_min
        area = w * h
        if 0 < area < 10000:  # tweak these thresholds for your objects
            bboxes.append((x_min, y_min, x_max, y_max))
        
    
    if len(bboxes) == 0:
        print(f"No bboxes generated for dir {save_dir}")
    
    bbox_output_file = save_dir / "generated_bboxes.txt"
    str_bboxes = [(str(x_min), str(y_min), str(x_max), str(y_max)) for (x_min, y_min, x_max, y_max) in bboxes]

    with open(bbox_output_file, "w") as file:
        for line in str_bboxes:
            file.write(" ".join(line) + "\n")
        
    return bboxes
  

def prepare_seg_image(path, type):    
    mask = Image.open(path).convert("L")
    
    if type == "floor":
        mask = ImageOps.invert(mask)
    
    mask = np.array(mask)
    mask = (mask > 0).astype(np.uint8) * 255
    return mask


# Save the mask and the predicted bbox
def save_data(save_dir, image, obj_mask, box_mask, bbox):
    # if the dir already exists, delete it
    if (data_dir / category / "geco2_mask").exists():
        shutil.rmtree(data_dir / category / "geco2_mask")
    
    os.makedirs(save_dir, exist_ok=True)

    np.save(save_dir / "image.npy", image)
    image_pil = Image.fromarray(image, mode="RGBA")
    image_pil.save(save_dir / "image.png")

    np.save(save_dir / "obj_mask.npy", obj_mask)
    obj_mask_pil = Image.fromarray(obj_mask, mode="L")
    obj_mask_pil.save(save_dir / "obj_mask.png")
    
    np.save(save_dir / "box_mask.npy", box_mask)
    box_mask_pil = Image.fromarray(box_mask, mode="L")
    box_mask_pil.save(save_dir / "box_mask.png")

    np.save(save_dir / "bbox.npy", bbox)
    

# Extract output sam2 mask and best predicted bbox
def extract_mask_and_bbox(outputs, masks_tensor, image):
    masks = np.array(masks_tensor.detach().cpu().numpy())
    scores = np.array(outputs[0]['scores'].detach().cpu().numpy())[0]
    
    N = min(len(scores), masks.shape[0])
    scores = scores[:N]
    masks = masks[:N, :, :]
    
    max_scores_ind = np.argmax(scores, axis=-1).item()

    # max_ind = np.argmax(np.array(outputs[0]['scores']), axis=-1).item()
    # top5_inds = np.argsort(scores[0])[-5:][::-1]

    mask = masks[max_scores_ind, :, :].astype(np.uint8) * 255

    bboxes_tensor = outputs[0]['pred_boxes'].squeeze(0).detach().cpu().numpy()
    bbox_norm = np.array(bboxes_tensor[max_scores_ind])

    H, W = image.shape[:2]
    x1 = int(bbox_norm[0] * W)
    y1 = int(bbox_norm[1] * H)
    x2 = int(bbox_norm[2] * W)
    y2 = int(bbox_norm[3] * H)

    bbox_px = np.array([x1, y1, x2, y2], dtype=np.int32)

    return mask, bbox_px


def get_last_common_frame(obj_seg_dir, floor_seg_dir):
    obj_files = sorted(os.listdir(obj_seg_dir))
    floor_files = set(os.listdir(floor_seg_dir))  # O(1) lookup

    # Start from last object frame and walk backwards
    for f in reversed(obj_files):
        frame_id = f[-8:-4]  # "0010" from Objects_Mask0010.png
        candidate = f"Ground_Mask{frame_id}.png"
        if candidate in floor_files:
            return f, candidate, frame_id

    raise RuntimeError("No common frame found")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_args = get_argparser().parse_known_args()[0]
    model_args.zero_shot = False
    model = DataParallel(build_model(model_args).to(device))
    model.load_state_dict(torch.load('ext/GECO2/CNTQG_multitrain_ca44.pth', weights_only=True)['model'], strict=False)
    model.eval()
    
    num_empty_bboxes = 0
    start_time = time.time()
    data_dir = args.data_dir

    for category in os.listdir(data_dir):
        print(f"Processing category {category}")
        
        image_dir = data_dir / category / "images"
        obj_seg_dir = data_dir / category / "obj_seg"
        floor_seg_dir = data_dir / category / "floor_seg"
        
        if not os.path.exists(image_dir):
            print(f"No images found for category {category}")
            continue
        
        if not os.path.exists(obj_seg_dir):
            print(f"No object segmentations found for category {category}")
            continue
        
        if not os.path.exists(floor_seg_dir):
            print(f"No floor segmentations found for category {category}")
            continue
        
        # take 10th frame
        # TODO: get frame with best visibility of objects and container
        if os.path.exists(obj_seg_dir / "Objects_Mask0010.png") and os.path.exists(floor_seg_dir / "Ground_Mask0010.png") and os.path.exists(image_dir / "RGB0010.jpg"):
            image_frame = "RGB0010.jpg"
            obj_frame = "Objects_Mask0010.png"
            ground_frame = "Ground_Mask0010.png"
        else:
            # else take last available frame
            obj_frame, ground_frame, frame_id = get_last_common_frame(obj_seg_dir, floor_seg_dir)
            image_frame = "RGB" + frame_id + ".jpg"
        
        image_path = image_dir / image_frame
        obj_seg_path = obj_seg_dir / obj_frame
        floor_seg_path = floor_seg_dir / ground_frame

        image_rgba = np.array(Image.open(image_path).convert("RGBA"))
        obj_mask = prepare_seg_image(obj_seg_path, type="obj")
        box_mask = prepare_seg_image(floor_seg_path, type="floor")
        
        bboxes = get_bounding_boxes_from_mask(obj_mask, data_dir / category)

        bboxes_file_path = data_dir / category / "generated_bboxes.txt"
        points = []

        with open(bboxes_file_path, "r") as file:
            for line in file:
                points.append([int(n) for n in line.strip().split(' ')])
            
        if len(points) == 0:
            print(f"No points found for category {category}, skipping")
            num_empty_bboxes = num_empty_bboxes + 1
            continue

        enable_mask = True
        threshold = 0.5        
        input = {
            'image': image_rgba[:,:,:-1],
            'points': points
        }

        try:
            out_img, n, masks_ori_tensor, image, outputs, masks, img, scale, drawn_boxes = initial_process(input, enable_mask, threshold)
        except:
            print(f"Error processing category {category}")
        
        single_obj_mask, single_obj_bbox = extract_mask_and_bbox(outputs, masks_ori_tensor, image)
        
        save_dir = data_dir / category / "geco2_data"
        save_data(save_dir, image_rgba, single_obj_mask, box_mask, single_obj_bbox)
    
    time_to_finish = time.time() - start_time
    
    print(f"Number of all categories: {len(os.listdir(data_dir))}")
    print(f"{num_empty_bboxes} categories with no bboxes found")
    print(f"Time to finish: {time_to_finish} seconds")