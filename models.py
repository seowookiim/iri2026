from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .pooling import AdaptiveSSRP_T, AttentiveStatisticsPooling, SSRP_B, SSRP_T


class LightweightCNNBackbone(nn.Module):
    """Paper-aligned lightweight CNN backbone that outputs (B, C, F, T)."""

    def __init__(self, channels: Sequence[int] = (32, 64, 128)):
        super().__init__()
        ch = [1, *[int(v) for v in channels]]
        layers = []
        for idx, (cin, cout) in enumerate(zip(ch[:-1], ch[1:]), start=1):
            layers.extend(
                [
                    nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=True),
                ]
            )
            # Average pooling after conv1 and conv2 only.
            if idx < len(ch) - 1:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
        self.features = nn.Sequential(*layers)
        self.out_channels = ch[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return self.features(x)


class DepthwiseSeparableConv2dA(nn.Module):
    """ACDNet DSConv style: DW -> BN -> ReLU -> PW -> BN."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size=(3, 3), padding=1):
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.pw = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            bias=False,
        )
        self.pw_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.dw_bn(x)
        x = torch.relu(x)
        x = self.pw(x)
        x = self.pw_bn(x)
        return x


class ACDNetMelBackbone(nn.Module):
    """
    ACDNet-style backbone adapted to mel-spectrogram input.
    Output shape: (B, C, F, T)
    """

    def __init__(self, base_channels: int = 8):
        super().__init__()
        c = int(base_channels)
        c1, c2, c3, c4, c5 = c * 8, c * 16, c * 32, c * 48, c * 64

        self.sfeb = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=(1, 9), stride=(1, 2), padding=(0, 4), bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c1, kernel_size=(1, 5), stride=(1, 2), padding=(0, 2), bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )

        self.tfeb = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DepthwiseSeparableConv2dA(c3, c4, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            DepthwiseSeparableConv2dA(c4, c5, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_channels = c5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.sfeb(x)
        x = self.tfeb(x)
        return x


def build_backbone(name: str) -> nn.Module:
    key = (name or "lightweight_cnn").strip().lower()
    if key in {"lightweight_cnn", "lwcnn", "cnn"}:
        return LightweightCNNBackbone()
    if key in {"acdnet", "acdnet_mel", "acdnet_style"}:
        return ACDNetMelBackbone()
    raise ValueError(f"Unknown backbone: {name}")


def build_pooling(
    pooling: str,
    ssrp_w: int,
    ssrp_k: int,
    ssrp_b_k: int,
    asp_attention_hidden: int,
    adaptive_ks: Sequence[int],
    adaptive_gate_hidden: int,
    adaptive_gate_input: str = "meanstd",
    adaptive_gate_norm: bool = True,
    adaptive_gate_dropout: float = 0.0,
    adaptive_temperature: float = 1.5,
    adaptive_learnable_temperature: bool = False,
    adaptive_alpha_floor: float = 0.0,
    adaptive_use_branch_calibration: bool = True,
    adaptive_return_mode: str = "z",
    return_alpha: bool = False,
    channels: int | None = None,
) -> nn.Module:
    key = (pooling or "ssrp_t").strip().lower()
    if key == "gap":
        return nn.AdaptiveAvgPool2d(1)
    if key == "gmp":
        return nn.AdaptiveMaxPool2d(1)
    if key == "asp":
        if channels is None:
            raise ValueError("AttentiveStatisticsPooling requires known channels, got None")
        return AttentiveStatisticsPooling(
            channels=int(channels),
            attention_hidden=int(asp_attention_hidden),
        )
    if key == "ssrp_t":
        return SSRP_T(W=ssrp_w, K=ssrp_k, out_mode="mean")
    if key == "ssrp_b":
        return SSRP_B(W=ssrp_w, K=ssrp_b_k)
    if key in {"adaptive_ssrp_t", "adaptive_ssrp"}:
        if channels is None:
            raise ValueError("AdaptiveSSRP_T requires known channels, got None")
        mode = adaptive_return_mode
        if return_alpha and mode == "z":
            mode = "alpha"
        return AdaptiveSSRP_T(
            channels=int(channels),
            W=ssrp_w,
            Ks=adaptive_ks,
            gate_hidden=adaptive_gate_hidden,
            gate_input=adaptive_gate_input,
            gate_norm=adaptive_gate_norm,
            gate_dropout=adaptive_gate_dropout,
            temperature=adaptive_temperature,
            learnable_temperature=adaptive_learnable_temperature,
            alpha_floor=adaptive_alpha_floor,
            use_branch_calibration=adaptive_use_branch_calibration,
            return_mode=mode,
            out_mode="mean",
        )
    raise ValueError(f"Unknown pooling: {pooling}")


class PoolingAudioClassifier(nn.Module):
    """
    Backbone + pooling + linear head.
    Backbone stays fixed as architecture; pooling module is swappable.
    """

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        pooling: str = "ssrp_t",
        ssrp_w: int = 4,
        ssrp_k: int = 12,
        ssrp_b_k: int = 12,
        asp_attention_hidden: int = 128,
        adaptive_ks: Sequence[int] = (4, 8, 12),
        adaptive_gate_hidden: int = 128,
        adaptive_gate_input: str = "meanstd",
        adaptive_gate_norm: bool = True,
        adaptive_gate_dropout: float = 0.0,
        adaptive_temperature: float = 1.5,
        adaptive_learnable_temperature: bool = False,
        adaptive_alpha_floor: float = 0.0,
        adaptive_use_branch_calibration: bool = True,
        adaptive_return_mode: str = "z",
        head_hidden: int = 128,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
        return_alpha: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.pooling_name = pooling
        self.backbone = build_backbone(backbone)
        self.pool = build_pooling(
            pooling=pooling,
            ssrp_w=ssrp_w,
            ssrp_k=ssrp_k,
            ssrp_b_k=ssrp_b_k,
            asp_attention_hidden=asp_attention_hidden,
            adaptive_ks=adaptive_ks,
            adaptive_gate_hidden=adaptive_gate_hidden,
            adaptive_gate_input=adaptive_gate_input,
            adaptive_gate_norm=adaptive_gate_norm,
            adaptive_gate_dropout=adaptive_gate_dropout,
            adaptive_temperature=adaptive_temperature,
            adaptive_learnable_temperature=adaptive_learnable_temperature,
            adaptive_alpha_floor=adaptive_alpha_floor,
            adaptive_use_branch_calibration=adaptive_use_branch_calibration,
            adaptive_return_mode=adaptive_return_mode,
            return_alpha=return_alpha,
            channels=getattr(self.backbone, "out_channels", None),
        )
        self.head = nn.Sequential(
            nn.LazyLinear(int(head_hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(int(head_hidden), int(num_classes)),
        )
        self.return_alpha = bool(return_alpha)
        self.last_feature_shape = None

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)
        self.last_feature_shape = tuple(feat.shape)
        pooled = self.pool(feat)

        alpha = None
        aux = None
        if isinstance(pooled, tuple):
            pooled, aux = pooled
            if isinstance(aux, dict):
                alpha = aux.get("alpha")
            else:
                alpha = aux
        if isinstance(self.pool, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
            pooled = pooled.squeeze(-1).squeeze(-1)

        logits = self.head(pooled)
        if self.return_alpha:
            return logits, aux if aux is not None else alpha
        return logits
