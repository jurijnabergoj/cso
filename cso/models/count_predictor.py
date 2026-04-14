import torch
import torch.nn as nn
from cfg.configs import ExperimentConfig
from cso.models.attention.cross_attention import CrossAttentionBlock

_SLAT_CROSS_DIM = 64   # hidden dim for slat cross-attention branch


class CountPredictor(nn.Module):
    def __init__(
        self,
        shape_latent_dim=4096 * 8,
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
        use_packing_factor_head=False,
        use_slat_cross_attn=False,
        use_image_in_pf_head=False,
    ):
        super().__init__()
        self.use_hybrid = use_hybrid
        self.use_shape_latent = use_shape_latent
        self.use_slat = use_slat
        self.use_stats_encoder = use_stats_encoder
        self.use_conv3d_encoder = use_conv3d_encoder
        self.use_image_encoder = use_image_encoder
        self.use_masked_image_encoder = use_masked_image_encoder
        self.use_packing_factor_head = use_packing_factor_head
        self.use_slat_cross_attn = use_slat_cross_attn
        self.use_image_in_pf_head = use_image_in_pf_head
        self.slat_dim = slat_dim
        self.use_cross_attn = use_cross_attn and use_shape_latent

        num_tokens = shape_latent_dim // slat_dim
        self.grid_size = round(num_tokens ** (1 / 3))

        # shape latent branch
        if self.use_shape_latent:
            if self.use_cross_attn:
                self.cross_attn = CrossAttentionBlock(d_model=d_model, h=h, d_ff=d_ff, dropout=dropout)
                self.attn_pool = nn.Linear(d_model, 1)
                self.linear_proj = nn.Linear(slat_dim, d_model)
            elif self.use_conv3d_encoder:
                in_ch = slat_dim * 2
                self.shape_encoder = nn.Sequential(
                    nn.Conv3d(in_ch, 32, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 32), nn.ReLU(),
                    nn.Conv3d(32, 64, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 64), nn.ReLU(),
                    nn.MaxPool3d(2),
                    nn.Conv3d(64, 64, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 64), nn.ReLU(),
                    nn.AdaptiveAvgPool3d(1), nn.Flatten(),
                    nn.Linear(64, d_model), nn.ReLU(),
                )
            elif self.use_stats_encoder:
                stats_dim = 3 * slat_dim * 2
                self.shape_encoder = nn.Sequential(
                    nn.Linear(stats_dim, 64), nn.LayerNorm(64), nn.ReLU(),
                    nn.Linear(64, d_model), nn.ReLU(),
                )
            else:
                self.shape_encoder = nn.Sequential(
                    nn.Linear(shape_latent_dim * 2, 512), nn.LayerNorm(512), nn.ReLU(),
                    nn.Dropout(0.1), nn.Linear(512, d_model), nn.ReLU(),
                )

        # slat pooling branch (for count head)
        if self.use_slat:
            self.slat_encoder = nn.Sequential(
                nn.Linear(slat_dim * 4, 128), nn.LayerNorm(128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
            )

        # slat cross-attention branch (geometric container-object relation)
        if self.use_slat_cross_attn:
            self.slat_seq_proj = nn.Linear(slat_dim, _SLAT_CROSS_DIM)
            self.slat_cross_attn_block = CrossAttentionBlock(
                d_model=_SLAT_CROSS_DIM, h=4, d_ff=_SLAT_CROSS_DIM * 2, dropout=dropout
            )
            self.slat_cross_attn_pool = nn.Linear(_SLAT_CROSS_DIM, 1)

        # image encoder branches
        if self.use_image_encoder:
            self.image_proj = nn.Sequential(
                nn.Linear(image_feat_dim, d_model), nn.LayerNorm(d_model), nn.ReLU(),
            )

        if self.use_masked_image_encoder:
            self.container_image_proj = nn.Sequential(
                nn.Linear(image_feat_dim, d_model), nn.LayerNorm(d_model), nn.ReLU(),
            )
            self.object_image_proj = nn.Sequential(
                nn.Linear(image_feat_dim, d_model), nn.LayerNorm(d_model), nn.ReLU(),
            )

        # packing factor head
        if self.use_packing_factor_head:
            pf_input_dim = slat_dim * 4
            if self.use_image_in_pf_head and self.use_masked_image_encoder:
                pf_input_dim += 2 * 64
                self.pf_cont_proj = nn.Sequential(nn.Linear(image_feat_dim, 64), nn.ReLU())
                self.pf_obj_proj = nn.Sequential(nn.Linear(image_feat_dim, 64), nn.ReLU())
            self.pf_head = nn.Sequential(
                nn.Linear(pf_input_dim, 64), nn.LayerNorm(64), nn.ReLU(),
                nn.Linear(64, 1), nn.Sigmoid(),
            )

        # count head
        input_dim = geometric_feature_dim
        if self.use_shape_latent:
            input_dim += d_model
        if self.use_slat:
            input_dim += 128
        if self.use_slat_cross_attn:
            input_dim += _SLAT_CROSS_DIM
        if use_hybrid:
            input_dim += 1
        if use_image_encoder:
            input_dim += d_model
        if use_masked_image_encoder:
            input_dim += 2 * d_model

        self.count_head = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.ReLU(),
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
        slat_seq_container=None,
        slat_seq_object=None,
    ):
        B = shape_latent_container.shape[0]
        G = self.grid_size
        C = self.slat_dim
        features = []

        if self.use_shape_latent:
            if self.use_cross_attn:
                query_down = shape_latent_object.view(B, 512, 8, C).mean(dim=2)
                memory_down = shape_latent_container.view(B, 512, 8, C).mean(dim=2)
                query_tokens = self.linear_proj(query_down)
                memory_tokens = self.linear_proj(memory_down)
                tokens = self.cross_attn(query_tokens, memory_tokens)
                weights = torch.softmax(self.attn_pool(tokens).float(), dim=1)
                features.append((tokens.float() * weights).sum(dim=1))
            elif self.use_conv3d_encoder:
                cont_vol = shape_latent_container.permute(0, 2, 1).view(B, C, G, G, G)
                obj_vol = shape_latent_object.permute(0, 2, 1).view(B, C, G, G, G)
                features.append(self.shape_encoder(torch.cat([cont_vol, obj_vol], dim=1)))
            elif self.use_stats_encoder:
                cont_f = shape_latent_container.float()
                obj_f = shape_latent_object.float()
                stats = torch.cat([
                    cont_f.mean(dim=1), cont_f.std(dim=1), cont_f.amax(dim=1),
                    obj_f.mean(dim=1), obj_f.std(dim=1), obj_f.amax(dim=1),
                ], dim=-1)
                features.append(self.shape_encoder(stats))
            else:
                combined_shape = torch.cat([
                    shape_latent_container.view(B, -1), shape_latent_object.view(B, -1),
                ], dim=-1)
                features.append(self.shape_encoder(combined_shape))

        if self.use_slat:
            combined_slat = torch.cat([slat_features_container, slat_features_object], dim=-1)
            features.append(self.slat_encoder(combined_slat))

        if (self.use_slat_cross_attn
                and slat_seq_container is not None
                and slat_seq_object is not None):
            cont_seq = self.slat_seq_proj(slat_seq_container.float())   # (B, K, 64)
            obj_seq = self.slat_seq_proj(slat_seq_object.float())       # (B, K, 64)
            cross_out = self.slat_cross_attn_block(obj_seq, cont_seq)   # (B, K, 64)
            weights = torch.softmax(self.slat_cross_attn_pool(cross_out.float()), dim=1)
            features.append((cross_out.float() * weights).sum(dim=1))  # (B, 64)

        features.append(geometric_features)

        if self.use_hybrid and geometric_estimate is not None:
            features.append(torch.log1p(geometric_estimate.clamp(min=0.0)).unsqueeze(-1))

        if self.use_image_encoder:
            if image_feats is None:
                raise ValueError("use_image_encoder=True but image_feats is None.")
            features.append(self.image_proj(image_feats.float()))

        if self.use_masked_image_encoder:
            if container_image_feats is None or object_image_feats is None:
                raise ValueError(
                    "use_masked_image_encoder=True but container/object image feats are None."
                )
            features.append(self.container_image_proj(container_image_feats.float()))
            features.append(self.object_image_proj(object_image_feats.float()))

        combined = torch.cat(features, dim=-1)
        count = self.final_activation(self.count_head(combined)).squeeze(-1)

        if self.use_packing_factor_head:
            combined_slat_raw = torch.cat(
                [slat_features_container, slat_features_object], dim=-1
            )
            if (self.use_image_in_pf_head
                    and container_image_feats is not None
                    and object_image_feats is not None):
                pf_input = torch.cat([
                    combined_slat_raw.float(),
                    self.pf_cont_proj(container_image_feats.float()),
                    self.pf_obj_proj(object_image_feats.float()),
                ], dim=-1)
            else:
                pf_input = combined_slat_raw.float()
            pf_pred = self.pf_head(pf_input).squeeze(-1)
            return count, pf_pred

        return count


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
        use_packing_factor_head=cfg.ablation.use_packing_factor_head,
        use_slat_cross_attn=cfg.ablation.use_slat_cross_attn,
        use_image_in_pf_head=cfg.ablation.use_image_in_pf_head,
    )
