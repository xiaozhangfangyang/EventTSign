from __future__ import annotations

import torch
import torch.nn as nn

from .backbones import ConvNeXtV2Backbone
from .segment_hypergraph_branch import SegmentHypergraphBranch


def _cfg_get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class EventSignModel(nn.Module):
    """Hyper-TSign: ConvNeXtV2 Nano, CPM tokens, and geometry-aware hypergraphs.

    Input: event images [B, 3, H, W] in [0, 1], with red/blue event polarities.
    Output: unnormalized classification logits [B, num_classes].
    """

    def __init__(self, cfg):
        super().__init__()
        model_cfg = _cfg_get(cfg, "model")
        if model_cfg is None:
            raise ValueError("Configuration must contain a 'model' section.")

        # These choices define the released main model. Validate older configs
        # explicitly so an unsupported architecture is never selected silently.
        fixed_choices = {
            "backbone": "convnextv2_nano",
            "input_channels": 3,
            "enable_event_filter": False,
            "use_event_filter": False,
            "enable_hypergraph": False,
            "use_hg_filter": False,
            "use_event_binary_prior": False,
            "use_segment_hg_branch": True,
            "segment_use_cpm": True,
            "segment_use_gahn": True,
            "segment_reasoning_type": "gahn",
            "segment_use_proximity": True,
            "segment_use_orientation": True,
            "segment_use_semantic_affinity": True,
            "segment_relation_specific": False,
            "segment_membership_normalization": "edge_softmax",
            "segment_feature_source": "stage3_feat",
            "segment_response_source": "raw",
            "segment_raw_response_mode": "event_activity",
            "segment_token_use_response": True,
            "segment_geometry_mode": "legacy",
            "segment_geom_dim": 11,
            "segment_use_global_context": False,
            "segment_hg_fusion": "residual",
        }
        for key, expected in fixed_choices.items():
            value = _cfg_get(model_cfg, key, expected)
            if value != expected:
                raise ValueError(f"The main model requires model.{key}={expected!r}, got {value!r}.")

        self.backbone = ConvNeXtV2Backbone(
            pretrained=bool(_cfg_get(model_cfg, "pretrained", True)),
            pretrained_path=_cfg_get(model_cfg, "pretrained_path", None),
            drop_path_rate=float(_cfg_get(model_cfg, "drop_path_rate", 0.1)),
        )
        segment_dim = int(_cfg_get(model_cfg, "segment_hg_hidden_dim", 128))
        num_tokens = int(_cfg_get(model_cfg, "segment_num_tokens", 16))
        self.segment_branch = SegmentHypergraphBranch(
            feat_channels=self.backbone.feature_dims["stage3_feat"],
            out_dim=segment_dim,
            num_tokens=num_tokens,
            token_dim=int(_cfg_get(model_cfg, "segment_token_dim", 64)),
            token_temperature=float(_cfg_get(model_cfg, "segment_token_temperature", 0.7)),
            num_relation_hyperedges=int(_cfg_get(model_cfg, "segment_num_relation_hyperedges", num_tokens)),
            proximity_membership_temperature=float(_cfg_get(model_cfg, "segment_proximity_membership_temperature", 0.5)),
            orientation_membership_temperature=float(_cfg_get(model_cfg, "segment_orientation_membership_temperature", 0.5)),
            segment_feat_dim=int(_cfg_get(model_cfg, "segment_feat_dim", 64)),
            segment_hg_hidden_dim=segment_dim,
            segment_hg_num_layers=int(_cfg_get(model_cfg, "segment_hg_num_layers", 1)),
            distance_scale=float(_cfg_get(model_cfg, "segment_hg_distance_scale", 0.28)),
            angle_scale_deg=float(_cfg_get(model_cfg, "segment_hg_angle_scale", 25.0)),
            scale_ratio=float(_cfg_get(model_cfg, "segment_hg_scale_ratio", 3.0)),
        )
        self.segment_fusion = nn.Linear(segment_dim, self.backbone.out_dim)
        self.segment_fusion_scale = nn.Parameter(
            torch.tensor(float(_cfg_get(model_cfg, "segment_hg_fusion_scale_init", 0.05)))
        )
        self.classifier = nn.Linear(self.backbone.out_dim, int(_cfg_get(model_cfg, "num_classes", 58)))

    @staticmethod
    def _event_activity(x: torch.Tensor) -> torch.Tensor:
        red, green, blue = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        positive = (red - torch.maximum(green, blue)).clamp(0.0, 1.0)
        negative = (blue - torch.maximum(red, green)).clamp(0.0, 1.0)
        return torch.maximum(positive, negative)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.size(1) != 3:
            raise ValueError("Expected event images with shape [B, 3, H, W].")
        response = self._event_activity(x)
        feats = self.backbone(x)
        segment_feat = self.segment_branch(response, feats["stage3_feat"])
        pooled = feats["pooled"] + self.segment_fusion_scale * self.segment_fusion(segment_feat)
        return self.classifier(pooled)
