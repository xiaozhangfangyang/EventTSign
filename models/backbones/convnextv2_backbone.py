# ConvNeXt V2 components adapted from Meta's official implementation.
# See THIRD_PARTY_NOTICES.md and licenses/ConvNeXt-V2-LICENSE.txt.
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_

from eventsign.paths import resolve_project_path


class LayerNorm(nn.Module):
    """LayerNorm supporting channels_last and channels_first tensors."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        if data_format not in {"channels_last", "channels_first"}:
            raise ValueError(f"Unsupported data_format: {data_format}")
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.data_format = data_format

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class GRN(nn.Module):
    """Global Response Normalization from ConvNeXtV2."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class Block(nn.Module):
    """ConvNeXtV2 block adapted from the official implementation."""

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return identity + self.drop_path(x)


class ConvNeXtV2(nn.Module):
    """ConvNeXtV2 Nano parameters, retaining official checkpoint names."""

    def __init__(self, drop_path_rate: float = 0.1):
        super().__init__()
        depths = [2, 2, 8, 2]
        dims = [80, 160, 320, 640]
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        ))
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            self.stages.append(nn.Sequential(*[
                Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])
            ]))
            cur += depths[i]
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        # Retained solely for strict compatibility with existing checkpoints.
        # Hyper-TSign uses EventSignModel.classifier for prediction.
        self.head = nn.Linear(dims[-1], 1000)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(1.0)
        self.head.bias.data.mul_(1.0)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class ConvNeXtV2Backbone(nn.Module):
    """Nano backbone with explicit, validated local pretrained weights."""

    def __init__(
        self,
        pretrained: bool = True,
        pretrained_path: str | Path | None = None,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        if pretrained and not pretrained_path:
            raise ValueError("pretrained=true requires model.pretrained_path; use pretrained=false for random initialization.")
        if pretrained_path is not None:
            pretrained_path = resolve_project_path(pretrained_path)
        if pretrained and not Path(pretrained_path).is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_path}")
        self.name = "convnextv2_nano"
        self.model = ConvNeXtV2(drop_path_rate=drop_path_rate)
        # Preserve registered aliases used by previously trained main models.
        self.stem = self.model.downsample_layers[0]
        self.downsample_layers = self.model.downsample_layers
        self.stages = self.model.stages
        self.norm = self.model.norm
        self.out_dim = 640
        self.feature_dims = {"stage3_feat": 320, "feat_map": 640}
        self.pretrained_requested = bool(pretrained)
        self.pretrained_loaded = False
        self.pretrained_source = str(pretrained_path) if pretrained_path is not None else None
        self.pretrained_matched_tensors = 0
        if pretrained:
            self._load_pretrained_weights()

    @staticmethod
    def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model", "model_state_dict"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, dict):
            raise TypeError("Checkpoint is not a valid state_dict mapping.")
        cleaned = {}
        for key, value in checkpoint.items():
            if not isinstance(value, torch.Tensor):
                continue
            prefixes = ("module.", "backbone.", "model.", "encoder.")
            while key.startswith(prefixes):
                key = key.split(".", 1)[1]
            if not key.startswith("head."):
                cleaned[key] = value
        return cleaned

    def _load_pretrained_weights(self):
        checkpoint = torch.load(self.pretrained_source, map_location="cpu", weights_only=True)
        state_dict = self._extract_state_dict(checkpoint)
        model_state = self.model.state_dict()
        required = {key for key in model_state if not key.startswith("head.")}
        missing = sorted(required - state_dict.keys())
        mismatched = sorted(
            key for key in required & state_dict.keys()
            if state_dict[key].shape != model_state[key].shape
        )
        if missing or mismatched:
            raise RuntimeError(
                "Checkpoint does not contain a complete ConvNeXtV2 Nano backbone: "
                f"{len(missing)} missing and {len(mismatched)} shape-mismatched tensors. "
                f"First affected keys: {(missing + mismatched)[:5]}"
            )
        self.model.load_state_dict({key: state_dict[key] for key in required}, strict=False)
        self.pretrained_matched_tensors = len(required)
        self.pretrained_loaded = True

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shallow = self.stem(x)
        x = self.stages[0](shallow)
        stage3_feat = None
        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if i == 2:
                stage3_feat = x
        pooled = self.norm(x.mean(dim=(-2, -1)))
        return {"stage3_feat": stage3_feat, "pooled": pooled}
