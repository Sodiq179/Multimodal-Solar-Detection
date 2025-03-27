"""
Model definition for multimodal solar panel and boiler counting using EfficientNetV2 and metadata attention.
"""

import torch
import torch.nn as nn
import timm

class EfficientNetV2Meta(nn.Module):
    """
    A dual-path neural network that combines image features from EfficientNetV2 with metadata features
    processed through self-attention.

    Args:
        model_name (str): Name of EfficientNetV2 model variant.
        pretrained (bool): Whether to use pretrained weights.
        num_outputs (int): Number of regression targets (default is 2).
    """
    def __init__(self, model_name='efficientnetv2_b3', pretrained=True, num_outputs=2):
        super(EfficientNetV2Meta, self).__init__()

        # Image encoder
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        backbone_out = self.backbone.num_features

        # Metadata processor
        self.meta_fc = nn.Sequential(
            nn.Linear(5, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )

        # Multi-head self-attention on metadata
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

        # Final regressor
        self.regressor = nn.Sequential(
            nn.Linear(backbone_out + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_outputs),
            nn.Softplus()  # Ensures non-negative output
        )

    def forward(self, image, metadata):
        """
        Forward pass through the model.

        Args:
            image (torch.Tensor): Input image tensor (B, C, H, W)
            metadata (torch.Tensor): Metadata tensor (B, 5)

        Returns:
            torch.Tensor: Predicted panel and boiler counts
        """
        image_features = self.backbone(image)  # (B, N)

        meta_encoded = self.meta_fc(metadata).unsqueeze(1)  # (B, 1, 64)
        attended_meta, _ = self.attention(meta_encoded, meta_encoded, meta_encoded)  # (B, 1, 64)
        attended_meta = attended_meta.squeeze(1)  # (B, 64)

        combined = torch.cat([image_features, attended_meta], dim=1)
        out = self.regressor(combined)
        return out
