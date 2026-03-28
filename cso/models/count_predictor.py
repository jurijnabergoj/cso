import torch
import torch.nn as nn


class SAM3DCountPredictor(nn.Module):

    def __init__(
            self,
            shape_latent_dim=8 * 4096,
            slat_dim=8,
            geometric_feature_dim=13,
            use_hybrid=False
    ):
        super().__init__()
        self.use_hybrid = use_hybrid

        self.shape_encoder = nn.Sequential(
            nn.Linear(shape_latent_dim * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.slat_encoder = nn.Sequential(
            nn.Linear(slat_dim * 4, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Count prediction head
        input_dim = 256 + 128 + geometric_feature_dim
        if use_hybrid:
            input_dim += 1  # Add geometric estimate

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
            nn.ReLU()
        )

        if use_hybrid:
            # Output is a correction factor
            self.final_activation = nn.Sigmoid()  # WIll scale to [0.5, 1.5]
        else:
            # Output is a direct count
            self.final_activation = nn.ReLU()

    def forward(
            self,
            shape_latent_container,
            shape_latent_object,
            slat_features_container,
            slat_features_object,
            geometric_features,
            geometric_estimate=None
    ):
        """
        Args:
            shape_latent_container: (B, 4096, 8)
            shape_latent_object: (B, 4096, 8)
            slat_features_container: (B, 16)
            slat_features_object: (B, 16)
            geometric_features: (B, feature_dim)
            geometric_estimate: (B,) - baseline count estimate
        """

        B = shape_latent_container.shape[0]

        shape_container_flat = shape_latent_container.view(B, -1)
        shape_object_flat = shape_latent_object.view(B, -1)
        
        combined_shape = torch.cat([shape_container_flat, shape_object_flat], dim=-1)
        shape_feat = self.shape_encoder(combined_shape)

        combined_slat = torch.cat([slat_features_container, slat_features_object], dim=-1) # (B, 32)
        slat_feat = self.slat_encoder(combined_slat)

        # Combine all features
        features = [shape_feat, slat_feat, geometric_features]
        
        if self.use_hybrid and geometric_estimate is not None:
            features.append(geometric_estimate.unsqueeze(-1))

        combined = torch.cat(features, dim=-1)

        # Predict
        output = self.count_head(combined)
        output = self.final_activation(output).squeeze(-1)

        if self.use_hybrid and geometric_estimate is not None:
            # Output is correction factor in [0.5, 1.5]
            correction = output * 1.0 + 0.5
            final_count = geometric_estimate * correction
            return final_count, correction
        else:
            return output, None
