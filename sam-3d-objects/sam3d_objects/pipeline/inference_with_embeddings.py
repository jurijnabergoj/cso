import torch
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
from typing import List, Union
from PIL import Image
import numpy as np
import trimesh
from pytorch3d.transforms import Transform3d, quaternion_to_matrix, matrix_to_quaternion
import os
import sys
from typing import Union, Optional, List, Callable
import numpy as np
from PIL import Image
from omegaconf import OmegaConf, DictConfig, ListConfig
from hydra.utils import instantiate, get_method
import torch
import math
import utils3d
import shutil
import subprocess
import seaborn as sns
from PIL import Image
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from copy import deepcopy
from kaolin.visualize import IpyTurntableVisualizer
from kaolin.render.camera import Camera, CameraExtrinsics, PinholeIntrinsics
import builtins
from pytorch3d.transforms import quaternion_multiply, quaternion_invert

import sam3d_objects  # REMARK(Pierre) : do not remove this import
from sam3d_objects.pipeline.inference_pipeline_pointmap import InferencePipelinePointMap
from sam3d_objects.model.backbone.tdfy_dit.utils import render_utils

from sam3d_objects.utils.visualization import SceneVisualizer

__all__ = ["Inference"]

WHITELIST_FILTERS = [
    lambda target: target.split(".", 1)[0] in {"sam3d_objects", "torch", "torchvision", "moge"},
]

BLACKLIST_FILTERS = [
    lambda target: get_method(target)
    in {
        builtins.exec,
        builtins.eval,
        builtins.__import__,
        os.kill,
        os.system,
        os.putenv,
        os.remove,
        os.removedirs,
        os.rmdir,
        os.fchdir,
        os.setuid,
        os.fork,
        os.forkpty,
        os.killpg,
        os.rename,
        os.renames,
        os.truncate,
        os.replace,
        os.unlink,
        os.fchmod,
        os.fchown,
        os.chmod,
        os.chown,
        os.chroot,
        os.fchdir,
        os.lchown,
        os.getcwd,
        os.chdir,
        shutil.rmtree,
        shutil.move,
        shutil.chown,
        subprocess.Popen,
        builtins.help,
    },
]


class InferenceWithEmbeddings:
    # public facing inference API
    # only put publicly exposed arguments here
    def __init__(self, config_file: str, compile: bool = False):
        # load inference pipeline
        config = OmegaConf.load(config_file)
        config.rendering_engine = "pytorch3d"  # overwrite to disable nvdiffrast
        config.compile_model = compile
        config.workspace_dir = os.path.dirname(config_file)
        check_hydra_safety(config, WHITELIST_FILTERS, BLACKLIST_FILTERS)
        self._pipeline: InferencePipelinePointMap = instantiate(config)

    def merge_mask_to_rgba(self, image, mask):
        mask = mask.astype(np.uint8) * 255
        mask = mask[..., None]
        # embed mask in alpha channel
        rgba_image = np.concatenate([image[..., :3], mask], axis=-1)
        return rgba_image

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray],
        mask: Optional[Union[None, Image.Image, np.ndarray]],
        seed: Optional[int] = None,
        pointmap=None,
    ) -> dict:
        image = self.merge_mask_to_rgba(image, mask)
        return self._pipeline.run(
            image,
            None,
            seed,
            stage1_only=False,
            with_mesh_postprocess=False,
            with_texture_baking=False,
            with_layout_postprocess=False,
            use_vertex_color=True,
            stage1_inference_steps=None,
            pointmap=pointmap,
        )
    
    def run_with_embeddings(
        self,
        image: Union[None, Image.Image, np.ndarray],
        mask: Union[None, Image.Image, np.ndarray] = None,
        seed=42,
        stage1_inference_steps=None,
        stage2_inference_steps=None,
    ) -> dict:
        """
        Parameters:
        - image (Image): The input image to be processed.
        - seed (int, optional): The random seed for reproducibility. Default is 42.
        - stage1_only (bool, optional): If True, only the sparse structure is sampled and returned. Default is False.
        - with_mesh_postprocess (bool, optional): If True, performs mesh post-processing. Default is True.
        - with_texture_baking (bool, optional): If True, applies texture baking to the 3D model. Default is True.
        Returns:
        - dict: A dictionary containing the GLB file and additional data from the sparse structure sampling.
        """

        
        image = self._pipeline.merge_image_and_mask(image, mask)
        with self._pipeline.device:
            pointmap_dict = self._pipeline.compute_pointmap(image)
            pointmap = pointmap_dict["pointmap"]

            ss_input_dict = self._pipeline.preprocess_image(image, self._pipeline.ss_preprocessor, pointmap=pointmap)
            
            slat_input_dict = self._pipeline.preprocess_image(image, self._pipeline.slat_preprocessor, pointmap=pointmap)
            torch.manual_seed(seed)

            ss_return_dict = self._sample_sparse_structure_with_embeddings(
                ss_input_dict,
                inference_steps=stage1_inference_steps,
            )
            ss_return_dict.update(self._pipeline.pose_decoder(ss_return_dict))

            if "scale" in ss_return_dict:
                ss_return_dict["scale"] = ss_return_dict["scale"] * ss_return_dict["downsample_factor"]

            coords = ss_return_dict["coords"]

            slat_dict = self._sample_slat_with_embeddings(
                slat_input_dict,
                coords,
                inference_steps=stage2_inference_steps,
            )

            outputs = self._pipeline.decode_slat(
                slat_dict['slat'], self._pipeline.decode_formats
            )
            
            outputs = self._pipeline.postprocess_slat_output(
                outputs, with_mesh_postprocess=False, with_texture_baking=False, use_vertex_color=True
            )

            return {
                **ss_return_dict,
                **slat_dict,
                **outputs,
            }
           
    def _sample_sparse_structure_with_embeddings(
            self,
            ss_input_dict: dict,
            inference_steps=None,
            use_distillation=False
    ):
        ss_generator = self._pipeline.models["ss_generator"]
        ss_decoder = self._pipeline.models["ss_decoder"]

        if use_distillation:
            ss_generator.no_shortcut = False
            ss_generator.reverse_fn.strength = 0
            ss_generator.reverse_fn.strength_pm = 0
        else:
            ss_generator.no_shortcut = True
            ss_generator.reverse_fn.strength = self._pipeline.ss_cfg_strength
            ss_generator.reverse_fn.strength_pm = self._pipeline.ss_cfg_strength_pm

        prev_inference_steps = ss_generator.inference_steps
        if inference_steps:
            ss_generator.inference_steps = inference_steps

        image = ss_input_dict["image"]
        bs = image.shape[0]

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self._pipeline.shape_model_dtype):
                if self._pipeline.is_mm_dit():
                    latent_shape_dict = {
                        k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                        for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                    }
                else:
                    latent_shape_dict = (bs,) + (4096, 8)

                condition_args, condition_kwargs = self._pipeline.get_condition_input(
                    self._pipeline.condition_embedders["ss_condition_embedder"],
                    ss_input_dict,
                    self._pipeline.ss_condition_input_mapping,
                )
                condition_embedding = condition_args[0] if condition_args else None

                return_dict = ss_generator(
                    latent_shape_dict,
                    image.device,
                    *condition_args,
                    **condition_kwargs,
                )
                if not self._pipeline.is_mm_dit():
                    return_dict = {"shape": return_dict}

                shape_latent = return_dict["shape"]
                ss = ss_decoder(
                    shape_latent.permute(0, 2, 1)
                    .contiguous()
                    .view(shape_latent.shape[0], 8, 16, 16, 16)
                )
                coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()

                return_dict['coords_original'] = coords
                original_shape = coords.shape
                if self._pipeline.downsample_ss_dist > 0:
                    from sam3d_objects.pipeline.inference_utils import prune_sparse_structure
                    coords = prune_sparse_structure(
                        coords,
                        max_neighbor_axes_dist=self._pipeline.downsample_ss_dist,
                    )
                from sam3d_objects.pipeline.inference_utils import downsample_sparse_structure
                coords, downsample_factor = downsample_sparse_structure(coords)

                return_dict["shape_latent"] = shape_latent  # (B, 4096, 8)
                return_dict["condition_embedding"] = condition_embedding  # from image encoder
                return_dict["coords"] = coords
                return_dict["downsample_factor"] = downsample_factor

        ss_generator.inference_steps = prev_inference_steps
        return return_dict

    def _sample_slat_with_embeddings(
            self,
            slat_input: dict,
            coords: torch.Tensor,
            inference_steps=None,
            use_distillation=False
    ):
        image = slat_input["image"]
        DEVICE = image.device
        slat_generator = self._pipeline.models["slat_generator"]
        latent_shape = (image.shape[0],) + (coords.shape[0], 8)

        prev_inference_steps = slat_generator.inference_steps
        if inference_steps:
            slat_generator.inference_steps = inference_steps
        if use_distillation:
            slat_generator.no_shortcut = False
            slat_generator.reverse_fn.strength = 0
        else:
            slat_generator.no_shortcut = True
            slat_generator.reverse_fn.strength = self._pipeline.slat_cfg_strength

        with torch.autocast(device_type="cuda", dtype=self._pipeline.dtype):
            with torch.no_grad():
                condition_args, condition_kwargs = self._pipeline.get_condition_input(
                    self._pipeline.condition_embedders["slat_condition_embedder"],
                    slat_input,
                    self._pipeline.slat_condition_input_mapping,
                )
                condition_args += (coords.cpu().numpy(),)

                slat_raw = slat_generator(
                    latent_shape, DEVICE, *condition_args, **condition_kwargs
                )
                slat = sp.SparseTensor(
                    coords=coords,
                    feats=slat_raw[0],
                ).to(DEVICE)
                slat = slat * self._pipeline.slat_std.to(DEVICE) + self._pipeline.slat_mean.to(DEVICE)

        slat_generator.inference_steps = prev_inference_steps

        return {
            'slat': slat,
            'slat_features': slat.feats,  # (N_points, 8)
            'coords': coords,
        }


def check_target(
    target: str,
    whitelist_filters: List[Callable],
    blacklist_filters: List[Callable],
):
    if any(filt(target) for filt in whitelist_filters):
        if not any(filt(target) for filt in blacklist_filters):
            return
    raise RuntimeError(
        f"target '{target}' is not allowed to be hydra instantiated, if this is a mistake, please do modify the whitelist_filters / blacklist_filters"
    )


def check_hydra_safety(
    config: DictConfig,
    whitelist_filters: List[Callable],
    blacklist_filters: List[Callable],
):
    to_check = [config]
    while len(to_check) > 0:
        node = to_check.pop()
        if isinstance(node, DictConfig):
            to_check.extend(list(node.values()))
            if "_target_" in node:
                check_target(node["_target_"], whitelist_filters, blacklist_filters)
        elif isinstance(node, ListConfig):
            to_check.extend(list(node))


def load_image(path):
    image = Image.open(path)
    image = np.array(image)
    image = image.astype(np.uint8)
    return image


def load_mask(path):
    print(path)
    mask = load_image(path)
    # mask = mask > 0
    # if mask.ndim == 3:
    #     mask = mask[..., -1]
    mask = mask[..., :3]
    mask = mask.sum(axis=-1) > 0
    mask = mask.astype(np.uint8)
    return mask


def load_single_mask(folder_path, index=0, extension=".png"):
    masks = load_masks(folder_path, [index], extension)
    return masks[0]


def load_masks(folder_path, indices_list=None, extension=".png"):
    print("loading masks")
    masks = []
    indices_list = [] if indices_list is None else list(indices_list)
    if not len(indices_list) > 0:  # get all all masks if not provided
        idx = 0
        while os.path.exists(os.path.join(folder_path, f"{idx}{extension}")):
            indices_list.append(idx)
            idx += 1

    for idx in indices_list:
        mask_path = os.path.join(folder_path, f"{idx}{extension}")
        assert os.path.exists(mask_path), f"Mask path {mask_path} does not exist"
        mask = load_mask(mask_path)
        masks.append(mask)
    return masks


def display_image(image, masks=None):
    def imshow(image, ax):
        ax.axis("off")
        ax.imshow(image)

    grid = (1, 1) if masks is None else (2, 2)
    fig, axes = plt.subplots(*grid)
    if masks is not None:
        mask_colors = sns.color_palette("husl", len(masks))
        black_image = np.zeros_like(image[..., :3], dtype=float)  # background
        mask_display = np.copy(black_image)
        mask_union = np.zeros_like(image[..., :3])
        for i, mask in enumerate(masks):
            mask_display[mask] = mask_colors[i]
            mask_union |= mask[..., None] if mask.ndim == 2 else mask
        imshow(black_image, axes[0, 1])
        imshow(mask_display, axes[1, 0])
        imshow(image * mask_union, axes[1, 1])

    image_axe = axes if masks is None else axes[0, 0]
    imshow(image, image_axe)

    fig.tight_layout(pad=0)
    fig.show()


def extract_geometric_features(outputs_container, outputs_object):
    features = {}

    # 1. Coordinate-based features (scale invariant)
    container_coords = outputs_container['coords'][:, 1:]  # Remove batch dim
    object_coords = outputs_object['coords'][:, 1:]

    container_bbox = container_coords.max(0)[0] - container_coords.min(0)[0]
    object_bbox = object_coords.max(0)[0] - object_coords.min(0)[0]
    container_bbox_vol = container_bbox.prod()
    object_bbox_vol = object_bbox.prod()

    features['bbox_vol_ratio'] = (object_bbox_vol / (container_bbox_vol + 1e-6)).item()

    # Point density
    features['container_point_density'] = container_coords.shape[0] / (container_bbox_vol.item() + 1e-6)
    features['object_point_density'] = object_coords.shape[0] / (object_bbox_vol.item() + 1e-6)
    features['density_ratio'] = features['object_point_density'] / (features['container_point_density'] + 1e-6)

    # Sparsity measure
    features['container_sparsity'] = container_coords.shape[0] / (64 ** 3)
    features['object_sparsity'] = object_coords.shape[0] / (64 ** 3)

    # 2. Translation and scale features
    features['translation_magnitude'] = outputs_object['translation'].norm().item()
    features['scale_ratio'] = (outputs_object['scale'] / (outputs_container['scale'] + 1e-6)).mean().item()

    # 3. Mesh-based features
    if outputs_container.get('glb') is not None and outputs_object.get('glb') is not None:
        container_mesh = outputs_container['glb']
        object_mesh = outputs_object['glb']

        features['mesh_vol_ratio'] = object_mesh.volume / (container_mesh.volume + 1e-6)
        features['mesh_sa_ratio'] = object_mesh.area / (container_mesh.area + 1e-6)

        try:
            object_convex = trimesh.convex.convex_hull(object_mesh)
            features['object_convexity'] = object_mesh.volume / (object_convex.volume + 1e-6)
        except:
            features['object_convexity'] = 1.0

        obj_dims = object_mesh.bounds[1] - object_mesh.bounds[0]
        features['object_aspect'] = obj_dims.max() / (obj_dims.min() + 1e-6)

        container_dims = container_mesh.bounds[1] - container_mesh.bounds[0]
        features['container_aspect'] = container_dims.max() / (container_dims.min() + 1e-6)
    else:
        features['mesh_vol_ratio'] = features['bbox_vol_ratio']
        features['mesh_sa_ratio'] = features['bbox_vol_ratio'] ** (2 / 3)
        features['object_convexity'] = 1.0
        features['object_aspect'] = 1.0
        features['container_aspect'] = 1.0

    # 4. Rotation features
    rot_matrix = quaternion_to_matrix(outputs_object['rotation'])
    features['rotation_determinant'] = torch.det(rot_matrix).item()

    features_list = [
        features['bbox_vol_ratio'],
        features['container_point_density'],
        features['object_point_density'],
        features['density_ratio'],
        features['container_sparsity'],
        features['object_sparsity'],
        features['translation_magnitude'],
        features['scale_ratio'],
        features['mesh_vol_ratio'],
        features['mesh_sa_ratio'],
        features['object_convexity'],
        features['object_aspect'],
        features['container_aspect'],
        features['rotation_determinant'],
        # Non-linear transformations
        np.log1p(features['bbox_vol_ratio']),
        np.sqrt(features['mesh_sa_ratio']),
        # Interaction terms
        features['bbox_vol_ratio'] * features['object_convexity'],
        features['density_ratio'] * features['scale_ratio'],
        features['container_sparsity'] / (features['object_sparsity'] + 1e-6),
        features['object_aspect'] / (features['container_aspect'] + 1e-6)
    ]

    return torch.tensor(features_list, dtype=torch.float32)


def compute_geometric_count_estimate(outputs_container, outputs_object, packing_factor):
    if outputs_container.get('glb') is not None and outputs_object.get('glb') is not None:
        container_vol = outputs_container['glb'].volume
        object_vol = outputs_object['glb'].volume
    else:
        # Use bounding box volumes
        container_coords = outputs_container['coords'][:, 1:]
        object_coords = outputs_object['coords'][:, 1:]
        container_bbox = container_coords.max(0)[0] - container_coords.min(0)[0]
        object_bbox = object_coords.max(0)[0] - object_coords.min(0)[0]
        container_vol = container_bbox.prod().item()
        object_vol = object_bbox.prod().item()

    geometric_estimate = (container_vol * packing_factor) / (object_vol + 1e-6)
    return max(1.0, geometric_estimate)
