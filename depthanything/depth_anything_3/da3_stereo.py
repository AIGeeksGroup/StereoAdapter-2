"""
Depth Anything 3 encoder for RAFT-Stereo.

This module wraps DA3's DinoV2 backbone and DPT head to provide
an interface compatible with RAFT-Stereo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

try:
    from depth_anything_3.api import DepthAnything3 as DA3Official
    HAS_OFFICIAL_API = True
except ImportError:
    HAS_OFFICIAL_API = False

from depthanything.depth_anything_3.model.dinov2.dinov2 import DinoV2
from depthanything.depth_anything_3.model.dpt import DPT


class DepthAnything3(nn.Module):
    """
    Depth Anything 3 encoder for stereo matching.

    This wraps DA3's DinoV2 backbone and DPT head to provide
    an interface compatible with RAFT-Stereo (similar to DAv2's DepthAnythingV2).

    Args:
        encoder: Encoder type ('vitb', 'vitl', 'vitg')
        features: DPT head feature dimension
        out_channels: DPT output channels per stage
        pretrained_path: Path to pretrained checkpoint (HuggingFace repo or local dir)
    """

    def __init__(
        self,
        encoder: str = 'vitb',
        features: int = 128,
        out_channels: List[int] = [96, 192, 384, 768],
        pretrained_path: Optional[str] = None,
    ):
        super().__init__()

        self.encoder = encoder

        embed_dim_map = {'vitb': 768, 'vitl': 1024, 'vitg': 1536}
        self.embed_dim = embed_dim_map[encoder]

        if pretrained_path is not None and HAS_OFFICIAL_API:
            logging.info(f"Loading DA3 using official API from {pretrained_path}")

            da3_official = DA3Official.from_pretrained(pretrained_path)

            self.pretrained = da3_official.model.backbone
            self.depth_head = da3_official.model.head
            self._use_official = True

            logging.info(f"DA3 loaded via official API: backbone + DualDPT head")

        else:
            if pretrained_path is not None:
                logging.warning("Official API not available, using manual module creation")

            out_layers_map = {
                'vitb': [5, 7, 9, 11],
                'vitl': [5, 11, 17, 23],
                'vitg': [9, 19, 29, 39],
            }

            self.pretrained = DinoV2(
                name=encoder,
                out_layers=out_layers_map[encoder],
                alt_start=4,
                qknorm_start=4,
                rope_start=4,
                cat_token=True,
            )

            dim_in = self.embed_dim * 2

            self.depth_head = DPT(
                dim_in=dim_in,
                features=features,
                out_channels=out_channels,
                patch_size=14,
                output_dim=1,
                use_sky_head=False,
            )
            self._use_official = False

    def _extract_multiscale_features_official(self, feats, H_pad, W_pad):
        """
        Extract multi-scale features from official DualDPT.
        Returns layer_*_rn (scratch layer outputs), consistent with DAv2.
        """
        head = self.depth_head
        patch_size = 14
        ph, pw = H_pad // patch_size, W_pad // patch_size

        B, S, N, C = feats[0][0].shape
        feats_processed = [feat[0].reshape(B * S, N, C) for feat in feats]

        resized_feats = []
        for stage_idx in range(4):
            x = feats_processed[stage_idx][:, 0:]
            x = head.norm(x)
            x = x.permute(0, 2, 1).reshape(B * S, C, ph, pw)
            x = head.projects[stage_idx](x)
            x = head.resize_layers[stage_idx](x)
            resized_feats.append(x)

        layer_1, layer_2, layer_3, layer_4 = resized_feats
        layer_1_rn = head.scratch.layer1_rn(layer_1)
        layer_2_rn = head.scratch.layer2_rn(layer_2)
        layer_3_rn = head.scratch.layer3_rn(layer_3)
        layer_4_rn = head.scratch.layer4_rn(layer_4)

        return [layer_1_rn, layer_2_rn, layer_3_rn, layer_4_rn]

    def forward(self, x: torch.Tensor, return_features: bool = False):
        """
        Forward pass through DA3 encoder.

        Args:
            x: Input image [B, 3, H, W]
            return_features: If True, return multi-scale features for RAFT-Stereo

        Returns:
            If return_features=True: List of 4 feature maps [H/4, H/8, H/16, H/32]
            Otherwise: depth prediction [B, 1, H, W]
        """
        H, W = x.shape[-2], x.shape[-1]
        patch_size = 14

        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        H_pad, W_pad = x.shape[-2], x.shape[-1]

        x = x.unsqueeze(1)  # [B, 1, 3, H, W] for DA3

        feats, _ = self.pretrained(x)

        if return_features:
            if self._use_official:
                return self._extract_multiscale_features_official(feats, H_pad, W_pad)
            else:
                features = self.depth_head(feats, H_pad, W_pad, patch_start_idx=0, return_features=True)
                return features
        else:
            output = self.depth_head(feats, H_pad, W_pad, patch_start_idx=0)
            if self._use_official:
                depth = output['depth']
                depth = depth.squeeze(1).unsqueeze(1)
            else:
                depth = output
                if isinstance(depth, tuple):
                    depth = depth[0]
                if depth.shape[1] > 1:
                    depth = depth[:, :1]

            if pad_h > 0 or pad_w > 0:
                depth = depth[:, :, :H, :W]
            return depth
