import torch
import torch.nn as nn
from cfg.configs import ExperimentConfig
from cso.models.attention.cross_attention import CrossAttentionBlock


class CountPredictor(nn.Module):
    def __init__(
        self,
        shape_latent_dim=4096 * 8,  # 4096 tokens × 8 dims = 32768
        slat_dim=8,
        geometric_feature_dim=13,
        d_model=256,
        d_ff=128,
        h=4,
        dropout=0.1,
        use_hybrid=False,
        use_cross_attn=False,
        use_shape_latent=True,
        use_slat=True,
        use_stats_encoder=False,
        use_conv3d_encoder=False,
        use_image_encoder=False,
        image_feat_dim=2048,
        use_masked_image_encoder=False,
    ):
        super().__init__()
        self.use_hybrid = use_hybrid
        self.use_shape_latent = use_shape_latent
        self.use_slat = use_slat
        self.use_stats_encoder = use_stats_encoder
        self.use_conv3d_encoder = use_conv3d_encoder
        self.use_image_encoder = use_image_encoder
        self.use_masked_image_encoder = use_masked_image_encoder
        self.slat_dim = slat_dim
        # cross-attn only makes sense when shape latents are used
        self.use_cross_attn = use_cross_attn and use_shape_latent

        # shape_latent is (4096, 8) = (16^3, slat_dim), so grid_size=16
        num_tokens = shape_latent_dim // slat_dim  # 32768 // 8 = 4096
        self.grid_size = round(num_tokens ** (1 / 3))  # 16

        if self.use_shape_latent:
            if self.use_cross_attn:
                self.cross_attn = CrossAttentionBlock(
                    d_model=d_model, h=h, d_ff=d_ff, dropout=dropout
                )
                # Learned attention pooling: scores each token, takes weighted sum
                self.attn_pool = nn.Linear(d_model, 1)
                # Projects each raw 8-dim token to d_model after local avg pooling
                self.linear_proj = nn.Linear(slat_dim, d_model)

            elif self.use_conv3d_encoder:
                # 3D CNN over the (8, 16, 16, 16) spatial volume.
                # The shape_latent is a 16^3 voxel grid of 8-dim features — the same
                # layout the SAM3D decoder uses internally.
                # Concatenate container + object along the channel dim -> (16, 16^3).
                in_ch = slat_dim * 2  # 16 (container + object channels)
                self.shape_encoder = nn.Sequential(
                    nn.Conv3d(
                        in_ch, 32, kernel_size=3, padding=1
                    ),  # (B, 32, 16, 16, 16)
                    nn.GroupNorm(8, 32),
                    nn.ReLU(),
                    nn.Conv3d(32, 64, kernel_size=3, padding=1),  # (B, 64, 16, 16, 16)
                    nn.GroupNorm(8, 64),
                    nn.ReLU(),
                    nn.MaxPool3d(2),  # (B, 64,  8,  8,  8)
                    nn.Conv3d(64, 64, kernel_size=3, padding=1),  # (B, 64,  8,  8,  8)
                    nn.GroupNorm(8, 64),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool3d(1),  # (B, 64,  1,  1,  1)
                    nn.Flatten(),  # (B, 64)
                    nn.Linear(64, d_model),
                    nn.ReLU(),
                )

            elif self.use_stats_encoder:
                # Global mean + std + max over 4096 tokens per shape.
                stats_dim = 3 * slat_dim * 2  # 48 for default config
                self.shape_encoder = nn.Sequential(
                    nn.Linear(stats_dim, 64),
                    nn.LayerNorm(64),
                    nn.ReLU(),
                    nn.Linear(64, d_model),
                    nn.ReLU(),
                )

            else:
                # Flatten: (B, 65536) through a large linear.
                self.shape_encoder = nn.Sequential(
                    nn.Linear(shape_latent_dim * 2, 512),
                    nn.LayerNorm(512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, d_model),
                    nn.ReLU(),
                )

        if self.use_slat:
            self.slat_encoder = nn.Sequential(
                nn.Linear(slat_dim * 4, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
            )

        if self.use_image_encoder:
            # Whole image features
            # global embedding from ResNet50 avgpool or DINOv2 CLS
            self.image_proj = nn.Sequential(
                nn.Linear(image_feat_dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
            )

        if self.use_masked_image_encoder:
            # Object specific features
            # local embeddings for container and single object
            self.container_image_proj = nn.Sequential(
                nn.Linear(image_feat_dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
            )
            self.object_image_proj = nn.Sequential(
                nn.Linear(image_feat_dim, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
            )

        # Count prediction head
        input_dim = geometric_feature_dim
        if self.use_shape_latent:
            input_dim += d_model
        if self.use_slat:
            input_dim += 128
        if use_hybrid:
            input_dim += 1  # log1p(geom_estimate)
        if use_image_encoder:
            input_dim += d_model
        if use_masked_image_encoder:
            input_dim += 2 * d_model  # container + object projections

        self.count_head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

        self.final_activation = nn.ReLU()

    def forward(
        self,
        shape_latent_container,
        shape_latent_object,
        slat_features_container,
        slat_features_object,
        geometric_features,
        geometric_estimate=None,
        image_feats=None,
        container_image_feats=None,
        object_image_feats=None,
    ):
        """
        Args:
            shape_latent_container: (B, 4096, 8)
            shape_latent_object: (B, 4096, 8)
            slat_features_container: (B, 16)
            slat_features_object: (B, 16)
            geometric_features: (B, feature_dim)
            geometric_estimate: (B,) - baseline count estimate
            image_feats: (B, image_feat_dim) - scene-level image features, optional
            container_image_feats: (B, image_feat_dim) - masked container crop features, optional
            object_image_feats: (B, image_feat_dim) - masked object crop features, optional
        """
        B = shape_latent_container.shape[0]
        G = self.grid_size  # 16
        C = self.slat_dim  # 8
        features = []

        if self.use_shape_latent:
            if self.use_cross_attn:
                # Local avg pool: average groups of 8 consecutive tokens before projecting.
                query_down = shape_latent_object.view(B, 512, 8, C).mean(
                    dim=2
                )  # (B, 512, 8)
                memory_down = shape_latent_container.view(B, 512, 8, C).mean(
                    dim=2
                )  # (B, 512, 8)
                query_tokens = self.linear_proj(query_down)  # (B, 512, d_model)
                memory_tokens = self.linear_proj(memory_down)  # (B, 512, d_model)

                tokens = self.cross_attn(
                    query_tokens, memory_tokens
                )  # (B, 512, d_model)

                weights = torch.softmax(
                    self.attn_pool(tokens).float(), dim=1
                )  # (B, 512, 1)
                pooled_tokens = (tokens.float() * weights).sum(dim=1)  # (B, d_model)
                features.append(pooled_tokens)

            elif self.use_conv3d_encoder:
                # Reshape (B, G^3, C) → (B, C, G, G, G) then concatenate container + object
                # along the channel dim to form a (B, 2C, G, G, G) volume for the 3D CNN.
                cont_vol = shape_latent_container.permute(0, 2, 1).view(B, C, G, G, G)
                obj_vol = shape_latent_object.permute(0, 2, 1).view(B, C, G, G, G)
                vol = torch.cat([cont_vol, obj_vol], dim=1)  # (B, 2C, G, G, G)
                features.append(self.shape_encoder(vol))  # (B, d_model)

            elif self.use_stats_encoder:
                # Global mean/std/max over token dim
                cont_f = shape_latent_container.float()
                obj_f = shape_latent_object.float()
                stats = torch.cat(
                    [
                        cont_f.mean(dim=1),
                        cont_f.std(dim=1),
                        cont_f.amax(dim=1),
                        obj_f.mean(dim=1),
                        obj_f.std(dim=1),
                        obj_f.amax(dim=1),
                    ],
                    dim=-1,
                )  # (B, 48)
                features.append(self.shape_encoder(stats))

            else:
                # Flatten both shapes and concatenate: (B, 2 * shape_latent_dim)
                combined_shape = torch.cat(
                    [
                        shape_latent_container.view(B, -1),
                        shape_latent_object.view(B, -1),
                    ],
                    dim=-1,
                )
                features.append(self.shape_encoder(combined_shape))  # (B, d_model)

        if self.use_slat:
            combined_slat = torch.cat(
                [slat_features_container, slat_features_object], dim=-1
            )
            features.append(self.slat_encoder(combined_slat))  # (B, 128)

        features.append(geometric_features)

        if self.use_hybrid and geometric_estimate is not None:
            # log1p to handle extreme values
            features.append(
                torch.log1p(geometric_estimate.clamp(min=0.0)).unsqueeze(-1)
            )

        if self.use_image_encoder:
            if image_feats is None:
                raise ValueError(
                    "use_image_encoder=True but image_feats is None. "
                    "Run scripts/generate_image_embeddings.py first."
                )
            features.append(self.image_proj(image_feats.float()))

        if self.use_masked_image_encoder:
            if container_image_feats is None or object_image_feats is None:
                raise ValueError(
                    "use_masked_image_encoder=True but container/object image feats are None. "
                    "Run: generate_image_embeddings.py --masked first."
                )
            features.append(self.container_image_proj(container_image_feats.float()))
            features.append(self.object_image_proj(object_image_feats.float()))

        combined = torch.cat(features, dim=-1)
        return self.final_activation(self.count_head(combined)).squeeze(-1)


def build_model(cfg: ExperimentConfig):
    return CountPredictor(
        shape_latent_dim=cfg.model.shape_latent_dim,
        slat_dim=cfg.model.slat_dim,
        geometric_feature_dim=cfg.model.geometric_feature_dim,
        d_model=cfg.model.d_model,
        d_ff=cfg.model.d_ff,
        h=cfg.model.h,
        dropout=cfg.model.dropout,
        use_hybrid=cfg.model.use_hybrid,
        use_cross_attn=cfg.ablation.use_attention,
        use_shape_latent=cfg.ablation.use_shape_latent,
        use_slat=cfg.ablation.use_slat,
        use_stats_encoder=cfg.ablation.use_stats_encoder,
        use_conv3d_encoder=cfg.ablation.use_conv3d_encoder,
        use_image_encoder=cfg.ablation.use_image_encoder,
        image_feat_dim=cfg.model.image_feat_dim,
        use_masked_image_encoder=cfg.ablation.use_masked_image_encoder,
    )
