from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal


class AttentiveStatisticsPooling(nn.Module):
    """
    Paper-style attentive statistics pooling over frame-level features.

    Input:
        x: (B, C, F, T)

    Output:
        pooled: (B, 2C), concatenating attentive mean and std.

    Notes:
        The original ASP paper defines attention over frame-level features h_t.
        For a CNN feature map (B, C, F, T), we first reduce frequency by mean
        to obtain frame-level descriptors of shape (B, C, T), and then apply
        temporal attention to compute weighted mean and weighted std.
    """

    def __init__(self, channels: int, attention_hidden: int = 128, eps: float = 1e-8):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be >= 1")
        if attention_hidden < 1:
            raise ValueError("attention_hidden must be >= 1")
        self.channels = int(channels)
        self.attention_hidden = int(attention_hidden)
        self.eps = float(eps)

        self.attention = nn.Sequential(
            nn.Conv1d(self.channels, self.attention_hidden, kernel_size=1, bias=True),
            nn.Tanh(),
            nn.Conv1d(self.attention_hidden, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"AttentiveStatisticsPooling expects 4D tensor (B,C,F,T), got shape={tuple(x.shape)}"
            )
        if x.size(1) != self.channels:
            raise ValueError(f"Expected channels={self.channels}, but got C={int(x.size(1))}")
        if not x.is_floating_point():
            raise TypeError("Input x must be a floating-point tensor.")

        # Convert CNN feature map into frame-level descriptors h_t.
        frames = x.mean(dim=2)  # (B, C, T)

        logits = self.attention(frames)  # (B, 1, T)
        alpha = torch.softmax(logits, dim=-1)

        mean = (alpha * frames).sum(dim=-1)
        second = (alpha * (frames ** 2)).sum(dim=-1)
        var = (second - mean ** 2).clamp_min(self.eps)
        std = torch.sqrt(var)
        return torch.cat([mean, std], dim=1)


class SSRP_T(nn.Module):
    """
    Paper-aligned SSRP-T (Top-K Sparse Salient Region Pooling)

    Input:
        x: (B, C, F, T)

    Paper steps:
        1) For each (B, C, F), compute moving mean over time with window size W
           -> wmean: (B, C, F, T-W+1)
        2) For each (B, C, F), select Top-K window means along time
        3) Average the Top-K values
           -> z_cf: (B, C, F)

    Output:
        - default: frequency-aware descriptor (B, C, F)
        - optional reduction: (B, C) only if explicitly requested

    Notes:
        - This matches the paper's freq-wise SSRP-T behavior.
        - The paper's key point is to KEEP frequency awareness, so default
          output is (B, C, F), not (B, C).
    """

    def __init__(
        self,
        W: int = 4,
        K: int = 12,
        out_mode: Literal["freq", "mean"] = "mean",
        fallback_mode: Literal["mean_time", "error"] = "mean_time",
    ):
        super().__init__()

        if W < 1:
            raise ValueError("W must be >= 1")
        if K < 1:
            raise ValueError("K must be >= 1")
        if out_mode not in ("freq", "mean"):
            raise ValueError("out_mode must be one of {'freq', 'mean'}")
        if fallback_mode not in ("mean_time", "error"):
            raise ValueError("fallback_mode must be one of {'mean_time', 'error'}")

        self.W = int(W)
        self.K = int(K)
        self.out_mode = out_mode
        self.fallback_mode = fallback_mode

    def _reduce_output(self, z_cf: torch.Tensor) -> torch.Tensor:
        """
        z_cf: (B, C, F)
        """
        if self.out_mode == "freq":
            return z_cf
        return z_cf.mean(dim=2)  # (B, C), optional only

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"SSRP_T expects 4D tensor (B, C, F, T), got shape={tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise TypeError("Input x must be a floating-point tensor.")

        B, C, Freq, T = x.shape

        if T < self.W:
            if self.fallback_mode == "error":
                raise ValueError(
                    f"Temporal length T={T} is smaller than window size W={self.W}."
                )
            # Practical fallback:
            # frequency-aware temporal mean, still preserving (B, C, F)
            z_cf = x.mean(dim=-1)  # (B, C, F)
            return self._reduce_output(z_cf)

        # Step 1) moving mean over TIME only, for each (B,C,F)
        if self.W == 1:
            wmean = x  # (B, C, F, T)
        else:
            x_reshaped = x.contiguous().view(B * C * Freq, 1, T)
            wmean = F.avg_pool1d(
                x_reshaped,
                kernel_size=self.W,
                stride=1,
            )  # (B*C*F, 1, T-W+1)
            Tw = wmean.size(-1)
            wmean = wmean.view(B, C, Freq, Tw)  # (B, C, F, T-W+1)

        # Step 2) Top-K along temporal-window axis
        Tw = wmean.size(-1)
        K_eff = min(self.K, Tw)
        topk_vals, _ = torch.topk(wmean, k=K_eff, dim=-1)  # (B, C, F, K_eff)

        # Step 3) Average Top-K values
        z_cf = topk_vals.mean(dim=-1)  # (B, C, F)

        return self._reduce_output(z_cf)

class AdaptiveSSRP_T(nn.Module):
    """
    Frequency-aware Adaptive SSRP-T via K-mixture.

    Input:
        x: (B, C, F, T)

    Internal branch descriptor:
        Z: (B, C, F, num_k)

    Final output:
        - out_mode == "freq" -> z: (B, C, F)
        - out_mode == "mean" -> z: (B, C)

    return_mode:
        - "z": returns z
        - "alpha": returns (z, alpha)
        - "details": returns (z, {"alpha", "logits", "Z", "gate_input",
                                  "fallback_used", "temperature"})
    """

    def __init__(
        self,
        channels: int,
        W: int = 2,
        Ks: Sequence[int] = (2, 4, 6),
        gate_hidden: int = 128,
        gate_input: str = "meanstd",
        gate_norm: bool = True,
        gate_dropout: float = 0.0,
        temperature: float = 1.5,
        learnable_temperature: bool = False,
        alpha_floor: float = 0.0,
        use_branch_calibration: bool = True,
        eps: float = 1e-8,
        return_mode: str = "z",
        out_mode: Literal["freq", "mean"] = "mean",
    ):
        super().__init__()

        if channels < 1:
            raise ValueError("channels must be >= 1")
        if W < 1:
            raise ValueError("W must be >= 1")
        if gate_hidden < 1:
            raise ValueError("gate_hidden must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if gate_input not in ("mean", "meanstd", "meanstdmax"):
            raise ValueError("gate_input must be one of {'mean', 'meanstd', 'meanstdmax'}")
        if return_mode not in ("z", "alpha", "details"):
            raise ValueError("return_mode must be one of {'z', 'alpha', 'details'}")
        if out_mode not in ("freq", "mean"):
            raise ValueError("out_mode must be one of {'freq', 'mean'}")
        if not (0.0 <= gate_dropout < 1.0):
            raise ValueError("gate_dropout must be in [0, 1)")
        if not (0.0 <= alpha_floor < 1.0):
            raise ValueError("alpha_floor must be in [0, 1)")

        ks = tuple(sorted(set(int(k) for k in Ks)))
        if len(ks) == 0 or any(k < 1 for k in ks):
            raise ValueError("Ks must contain one or more positive integers")

        self.channels = int(channels)
        self.W = int(W)
        self.Ks = ks
        self.num_k = len(self.Ks)
        self.gate_hidden = int(gate_hidden)
        self.gate_input = gate_input
        self.gate_norm_enabled = bool(gate_norm)
        self.gate_dropout = float(gate_dropout)
        self.learnable_temperature = bool(learnable_temperature)
        self.alpha_floor = float(alpha_floor)
        self.use_branch_calibration = bool(use_branch_calibration)
        self.eps = float(eps)
        self.return_mode = return_mode
        self.out_mode = out_mode

        if gate_input == "mean":
            gate_in_dim = self.channels
        elif gate_input == "meanstd":
            gate_in_dim = self.channels * 2
        else:
            gate_in_dim = self.channels * 3

        self.gate_norm = nn.LayerNorm(gate_in_dim) if self.gate_norm_enabled else nn.Identity()

        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, self.gate_hidden),
            nn.GELU(),
            nn.Dropout(p=self.gate_dropout),
            nn.Linear(self.gate_hidden, self.num_k),
        )

        if self.learnable_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(float(temperature))))
        else:
            self.register_buffer("_fixed_temperature", torch.tensor(float(temperature)), persistent=False)

        if self.use_branch_calibration:
            raw_init = math.log(math.exp(1.0) - 1.0)
            self.branch_raw_scale = nn.Parameter(
                torch.full((1, self.channels, 1, self.num_k), raw_init)
            )
            self.branch_bias = nn.Parameter(torch.zeros(1, self.channels, 1, self.num_k))
        else:
            self.register_parameter("branch_raw_scale", None)
            self.register_parameter("branch_bias", None)

    def _check_input(self, x: torch.Tensor) -> None:
        if x.dim() != 4:
            raise ValueError(
                f"AdaptiveSSRP_T expects 4D tensor (B,C,F,T), got shape={tuple(x.shape)}"
            )
        _, C, _, _ = x.shape
        if C != self.channels:
            raise ValueError(f"Expected channels={self.channels}, but got C={C}")
        if not x.is_floating_point():
            raise TypeError("Input x must be a floating-point tensor.")

    def _current_temperature(self) -> torch.Tensor:
        if self.learnable_temperature:
            return self.log_temperature.exp().clamp(min=0.5, max=5.0)
        return self._fixed_temperature

    def _compute_gate_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gate input is sample-wise summary over (F, T), producing:
            (B, C) or concatenated variants.
        """
        g_mean = x.mean(dim=(2, 3))

        if self.gate_input == "mean":
            g_in = g_mean
        else:
            g_std = x.std(dim=(2, 3), unbiased=False)
            if self.gate_input == "meanstd":
                g_in = torch.cat([g_mean, g_std], dim=1)
            else:
                g_max = x.amax(dim=(2, 3))
                g_in = torch.cat([g_mean, g_std, g_max], dim=1)

        return self.gate_norm(g_in)

    def _compute_window_mean(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Returns:
            wmean: (B, C, F, Tw)
        """
        B, C, freq, T = x.shape

        if T < self.W:
            return None

        if self.W == 1:
            return x

        x_ = x.contiguous().view(B * C * freq, 1, T)
        wmean = F.avg_pool1d(x_, kernel_size=self.W, stride=1)
        Tw = wmean.size(-1)
        return wmean.view(B, C, freq, Tw)

    def _apply_branch_calibration(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z: (B, C, F, num_k)
        """
        if not self.use_branch_calibration:
            return Z
        scale = F.softplus(self.branch_raw_scale) + self.eps
        return Z * scale + self.branch_bias

    def _multi_k_pool(self, wmean: torch.Tensor) -> torch.Tensor:
        """
        Args:
            wmean: (B, C, F, Tw)

        Returns:
            Z: (B, C, F, num_k)
        """
        _, _, _, Tw = wmean.shape
        Kmax = min(max(self.Ks), Tw)

        # topk along temporal-window axis
        topk_vals, _ = torch.topk(wmean, k=Kmax, dim=-1)   # (B, C, F, Kmax)
        cumsum = topk_vals.cumsum(dim=-1)

        zs = []
        for K in self.Ks:
            Keff = min(K, Tw)
            z_cf = cumsum[..., Keff - 1] / float(Keff)      # (B, C, F)
            zs.append(z_cf)

        Z = torch.stack(zs, dim=-1)                         # (B, C, F, num_k)
        Z = self._apply_branch_calibration(Z)
        return Z

    def _fallback_pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fallback when T < W:
        replicate frequency-aware global pooled descriptor across branches.

        Returns:
            Z: (B, C, F, num_k)
        """
        z_base = x.mean(dim=-1)                             # (B, C, F)
        Z = z_base.unsqueeze(-1).repeat(1, 1, 1, self.num_k)
        Z = self._apply_branch_calibration(Z)
        return Z

    def _reduce_output(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, C, F)
        """
        if self.out_mode == "freq":
            return z
        return z.mean(dim=2)                               # (B, C)

    def forward(
        self, x: torch.Tensor
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        self._check_input(x)

        g_in = self._compute_gate_input(x)                  # (B, gate_in_dim)
        logits = self.gate(g_in)                            # (B, num_k)

        temp = self._current_temperature()
        alpha = F.softmax(logits / temp, dim=-1)            # (B, num_k)

        if self.alpha_floor > 0.0:
            alpha = (1.0 - self.alpha_floor) * alpha + self.alpha_floor / self.num_k

        wmean = self._compute_window_mean(x)
        fallback_used = wmean is None

        Z = self._fallback_pool(x) if fallback_used else self._multi_k_pool(wmean)
        # Z: (B, C, F, num_k)

        z_freq = (Z * alpha.unsqueeze(1).unsqueeze(2)).sum(dim=-1)   # (B, C, F)
        z = self._reduce_output(z_freq)

        if self.return_mode == "z":
            return z

        if self.return_mode == "alpha":
            return z, alpha

        return z, {
            "alpha": alpha,                                 # (B, num_k)
            "logits": logits,                               # (B, num_k)
            "Z": Z,                                         # (B, C, F, num_k)
            "z_freq": z_freq,                               # (B, C, F)
            "gate_input": g_in,                             # (B, gate_in_dim)
            "fallback_used": torch.tensor(
                fallback_used, device=x.device, dtype=torch.bool
            ),
            "temperature": (
                temp.detach()
                if isinstance(temp, torch.Tensor)
                else torch.tensor(float(temp), device=x.device)
            ),
        }

    @staticmethod
    def alpha_entropy(
        alpha: torch.Tensor,
        eps: float = 1e-8,
        normalized: bool = False,
    ) -> torch.Tensor:
        """
        alpha: (B, num_k)
        """
        ent = -(alpha * (alpha + eps).log()).sum(dim=-1)

        if normalized:
            num_k = alpha.size(-1)
            if num_k > 1:
                ent = ent / math.log(num_k)
            else:
                ent = torch.zeros_like(ent)

        return ent.mean()

    @staticmethod
    def alpha_dominant_ratio(alpha: torch.Tensor) -> torch.Tensor:
        """
        alpha: (B, num_k)
        """
        return alpha.max(dim=-1).values.mean()


class SSRP_B(nn.Module):
    """
    SSRP-B (2D variant).
    Input:  x in (B, C, F, T)
    Step1) window mean over (F,T) with kernel (W,W), stride=1
    Step2) top-K over flattened spatial windows for each (B,C)
    Step3) average Top-K -> (B, C)
    """

    def __init__(self, W: int = 4, K: int = 12):
        super().__init__()
        if W < 1:
            raise ValueError("W must be >= 1")
        if K < 1:
            raise ValueError("K must be >= 1")
        self.W = int(W)
        self.K = int(K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"SSRP_B expects 4D tensor (B,C,F,T), got shape={tuple(x.shape)}")

        B, C, freq, T = x.shape
        if self.W <= 1 or freq < self.W or T < self.W:
            return x.mean(dim=(2, 3))

        wmean = F.avg_pool2d(x, kernel_size=(self.W, self.W), stride=1)
        flat = wmean.view(B, C, -1)
        K_eff = min(self.K, flat.size(-1))
        topk_vals, _ = torch.topk(flat, k=K_eff, dim=-1)
        return topk_vals.mean(dim=-1)
