"""
GLASS Flow with Feynman-Kac (FK) Corrector (Image Editing Adapted).

Implements sequential Monte Carlo (SMC) sampling on top of the GLASS
transition kernel.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Any

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Core primitives (Supports multi-dimensional Image Tensors natively)
# ---------------------------------------------------------------------------

def format_batch_variable(t, x_t: Tensor) -> Tensor:
    t = torch.tensor(t, device=x_t.device, dtype=x_t.dtype)
    if t.ndim == 0:
        t = t.unsqueeze(0)
    if len(t) < x_t.shape[0]:
        assert len(t) == 1
        t = torch.ones(x_t.shape[0], device=x_t.device, dtype=x_t.dtype) * t
    return t


def mult_first_dim(x: Tensor, t: Tensor) -> Tensor:
    if t.ndim == 0:
        return t * x
    t = t.view(-1)
    if x.size(0) != t.size(0):
        raise ValueError("Size of t must match first dimension of x.")
    return x * t.view(-1, *([1] * (x.dim() - 1)))


# ---------------------------------------------------------------------------
# GlassFlow  (Unmodified logic, handles arbitrary shapes)
# ---------------------------------------------------------------------------

class GlassFlow(nn.Module):
    def __init__(
        self,
        fm_model: nn.Module,
        clip_val: float = 1e-8,
        t_min: float = 0.001,
        t_max: float = 0.999,
        eta_t_clip: float = 200.0,
    ):
        super().__init__()
        self.fm_model   = fm_model
        self.clip_val   = clip_val
        self.t_min      = t_min
        self.t_max      = t_max
        self.eta_t_clip = eta_t_clip

    # -- schedulers (linear path) ------------------------------------------
    def alpha_t(self, t: Tensor) -> Tensor:     return t
    def dot_alpha_t(self, t: Tensor) -> Tensor: return torch.ones_like(t)
    def sigma_t(self, t: Tensor) -> Tensor:     return 1 - t
    def dot_sigma_t(self, t: Tensor) -> Tensor: return -torch.ones_like(t)

    def g_t(self, t: Tensor) -> Tensor:
        return (self.sigma_t(t) / torch.clamp(self.alpha_t(t), min=self.clip_val)) ** 2

    def g_t_inv(self, inp: Tensor) -> Tensor:
        return 1.0 / (1.0 + torch.sqrt(inp))

    # -- denoiser  x̂₁ = D(x_t, t) -----------------------------------------
    def denoiser(self, x_t: Tensor, t, **kwargs) -> Tensor:
        t = format_batch_variable(t, x_t)
        velocity = self.fm_model(x_t=x_t, t=t, **kwargs)
        diff = (mult_first_dim(velocity, self.sigma_t(t))
                - mult_first_dim(x_t, self.dot_sigma_t(t)))
        denom = (self.dot_alpha_t(t) * self.sigma_t(t)
                 - self.dot_sigma_t(t) * self.alpha_t(t))
        return mult_first_dim(diff, 1.0 / torch.clamp(denom, min=self.clip_val))

    # -- internal helpers --------------------------------------------------
    def bar_alpha_s(self, s: Tensor, bar_alpha_final: float) -> Tensor:
        return bar_alpha_final * s

    def dot_bar_alpha_s(self, s: Tensor, bar_alpha_final: float) -> Tensor:
        return bar_alpha_final * torch.ones_like(s)

    def bar_sigma_s(self, s: Tensor, sigma_cond_final: float) -> Tensor:
        return s * sigma_cond_final + (1 - s)

    def dot_bar_sigma_s(self, s: Tensor, sigma_cond_final: float) -> Tensor:
        return torch.ones_like(s) * (sigma_cond_final - 1.0)

    def _stable_inv(self, M: Tensor) -> Tensor:
        return torch.linalg.inv(
            M + 0.0001 * self.clip_val * torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
        )

    def _glass_denoiser(
        self,
        mu_s: Tensor, Cov_s: Tensor,
        X_t: Tensor, bar_X_s: Tensor,
        dtype: torch.dtype, precdtype: torch.dtype,
        **kwargs,
    ) -> Tensor:
        inv_Cov_s = self._stable_inv(Cov_s)
        bproduct  = mu_s @ inv_Cov_s @ mu_s
        t_star    = self.g_t_inv(1.0 / torch.clamp(bproduct, min=self.clip_val))
        weights   = self.alpha_t(t_star) * (mu_s @ inv_Cov_s) / torch.clamp(bproduct, min=self.clip_val)
        
        # weights[0] and weights[1] are scalars/1D, effectively broadcast over Image shape via mult_first_dim natively
        suff_stat = mult_first_dim(X_t, weights[0]) + mult_first_dim(bar_X_s, weights[1])
        return self.denoiser(x_t=suff_stat.to(dtype), t=t_star.to(dtype), **kwargs).to(dtype=precdtype)

    # -- core transition sampler -------------------------------------------
    def sample_glass_transition(
        self,
        X_t: Tensor,
        t_start: Tensor,
        t_end: Tensor,
        corr_rho: float,
        n_steps: int,
        dtype: torch.dtype,
        device: torch.device,
        return_traj: bool = False,
        precdtype: torch.dtype = torch.float64,
        **kwargs,
    ):
        s_vec = torch.linspace(self.t_min, self.t_max, n_steps + 1, dtype=precdtype, device=device)

        alpha_t_start = self.alpha_t(t_start).to(precdtype)
        alpha_t_end   = self.alpha_t(t_end).to(precdtype)
        sigma_t_start = self.sigma_t(t_start).to(precdtype)
        sigma_t_end   = self.sigma_t(t_end).to(precdtype)

        bar_gamma   = corr_rho * sigma_t_end / torch.clamp(sigma_t_start, min=self.clip_val)
        bar_X_s     = bar_gamma * X_t + torch.randn_like(X_t)

        bar_alpha_final = alpha_t_end - bar_gamma * alpha_t_start
        bar_sigma_final = torch.sqrt(
            torch.clamp(sigma_t_end ** 2 * (1 - corr_rho ** 2), min=0.0)
        )

        bar_alpha_s     = self.bar_alpha_s(s_vec, bar_alpha_final)
        dot_bar_alpha_s = self.dot_bar_alpha_s(s_vec, bar_alpha_final)
        bar_sigma_s     = self.bar_sigma_s(s_vec, bar_sigma_final)
        dot_bar_sigma_s = self.dot_bar_sigma_s(s_vec, bar_sigma_final)

        w_1 = dot_bar_sigma_s / torch.clamp(bar_sigma_s, min=self.clip_val)
        w_2 = dot_bar_alpha_s - w_1 * bar_alpha_s
        w_3 = -w_1 * bar_gamma

        X_t    = X_t.to(dtype=precdtype)
        traj   = [bar_X_s.cpu().detach().float()] if return_traj else None

        for i in range(len(s_vec) - 1):
            mu_s  = torch.tensor(
                [alpha_t_start, bar_alpha_s[i] + bar_gamma * alpha_t_start],
                dtype=precdtype, device=device,
            )
            Cov_s = torch.tensor(
                [[sigma_t_start ** 2, bar_gamma * sigma_t_start ** 2],
                 [bar_gamma * sigma_t_start ** 2,
                  bar_sigma_s[i] ** 2 + bar_gamma ** 2 * sigma_t_start ** 2]],
                dtype=precdtype, device=device,
            )
            glass_den = self._glass_denoiser(mu_s, Cov_s, X_t, bar_X_s, dtype, precdtype, **kwargs)
            velocity  = mult_first_dim(bar_X_s, w_1[i]) + mult_first_dim(glass_den, w_2[i]) + mult_first_dim(X_t, w_3[i])
            bar_X_s   = bar_X_s + (s_vec[i + 1] - s_vec[i]) * velocity
            if traj is not None:
                traj.append(bar_X_s.cpu().detach())

        return traj if return_traj else bar_X_s

    # -- DDPM-matched correlation ------------------------------------------
    def get_ddpm_corr(self, t_start: Tensor, t_end: Tensor) -> Tensor:
        return (
            self.alpha_t(t_start) * self.sigma_t(t_end)
            / torch.clamp(self.alpha_t(t_end) * self.sigma_t(t_start), min=self.clip_val)
        )


# ---------------------------------------------------------------------------
# FK Corrector
# ---------------------------------------------------------------------------

class GlassFlowFK(nn.Module):
    def __init__(self, glass_flow: GlassFlow, resample_every: int = 1):
        super().__init__()
        self.glass_flow    = glass_flow
        self.resample_every = resample_every

    @torch.no_grad()
    def sample(
        self,
        K: int,
        reward_fn: Callable[[Tensor], Tensor],
        t_backbone: Tensor,
        n_inner_steps: int,
        corr_rho: float,
        device: torch.device,
        dtype: torch.dtype,
        x_init: Optional[Tensor] = None,
        return_log_weights: bool = False,
        return_intermediates: bool = False,
        verbose: bool = False,
        **glass_kwargs,
    ) -> Tensor | tuple[Tensor, Tensor]:
        
        t_backbone = t_backbone.to(device=device, dtype=dtype)
        n_steps    = len(t_backbone) - 1

        if x_init is not None:
            X = x_init.to(device=device, dtype=dtype)
            assert X.shape[0] == K, f"x_init batch dim {X.shape[0]} != K={K}"
        else:
            shape = getattr(self.glass_flow, "data_shape", None)
            if shape is None:
                raise ValueError("Provide x_init or set glass_flow.data_shape = (C, H, W).")
            X = torch.randn(K, *shape, device=device, dtype=dtype)

        log_weights   = torch.zeros(K, device=device, dtype=dtype)
        intermediates = [X.cpu().float()] if return_intermediates else None

        for idx in range(n_steps):
            t_start = t_backbone[idx]
            t_end   = t_backbone[idx + 1]

            # (a) Propagate
            X = self.glass_flow.sample_glass_transition(
                X_t=X,
                t_start=t_start,
                t_end=t_end,
                corr_rho=corr_rho,
                n_steps=n_inner_steps,
                dtype=dtype,
                device=device,
                **glass_kwargs,
            ).to(dtype=dtype)

            # (b) Weight: reward at denoised prediction
            x1_hat      = self.glass_flow.denoiser(X, t=t_end).to(dtype=dtype)
            log_r       = reward_fn(x1_hat)                  # (K,)
            log_weights = log_weights + log_r

            # (c) Resample
            if (idx + 1) % self.resample_every == 0:
                weights = torch.softmax(log_weights, dim=0)  # normalise
                if verbose:
                    ess = 1.0 / (weights ** 2).sum().item()
                    print(f"[GlassFlowFK] step {idx+1}/{n_steps}  "
                          f"ESS={ess:.1f}/{K}  "
                          f"log_w: min={log_weights.min():.2f} "
                          f"max={log_weights.max():.2f}")
                indices     = torch.multinomial(weights, K, replacement=True)
                X           = X[indices]
                log_weights = torch.zeros(K, device=device, dtype=dtype)

            if intermediates is not None:
                intermediates.append(X.cpu().float())

        out = [X]
        if return_intermediates:
            out.append(intermediates)
        if return_log_weights:
            out.append(log_weights)
        return out[0] if len(out) == 1 else tuple(out)


# ---------------------------------------------------------------------------
# Reward helpers (Strictly dependent on utils functions)
# ---------------------------------------------------------------------------

class ImageMultiPromptReward(nn.Module):
    """
    Log-reward adapter for image editing. 
    Strictly calls original `clip_semantic_loss.L_N()` from `utils/flowgrad_utils.py` 
    to obtain guidance values without writing any new loss implementation.
    """

    def __init__(self, clip_loss_list: List[Any], guidance_scale: float = 1.0):
        super().__init__()
        # clip_loss_list expects initialized instances of clip_semantic_loss
        self.clip_loss_list = clip_loss_list
        self.guidance_scale = guidance_scale

    def forward(self, x1_hat: Tensor) -> Tensor:
        """
        Args:
            x1_hat: Denoised images of shape (K, C, H, W).
        Returns:
            log_r: (K,) log-reward tensor.
        """
        total_log_r = torch.zeros(x1_hat.shape[0], device=x1_hat.device, dtype=x1_hat.dtype)
        for clip_loss in self.clip_loss_list:
            # 必须直接调用 utils 里原本的计算函数 (clip_semantic_loss_L_N returns shape (K,))
            loss_k = clip_loss.L_N(x1_hat)
            # Energy / Reward is essentially negative loss
            total_log_r = total_log_r - self.guidance_scale * loss_k
            
        return total_log_r