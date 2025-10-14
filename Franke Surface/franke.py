from __future__ import annotations
import math
from typing import Tuple
import numpy as np
import torch
from torch import Tensor

def franke_np(xy: np.ndarray) -> np.ndarray:
    x, y = xy[:, 0], xy[:, 1]
    t1 = 0.75 * np.exp(-(9*x - 2)**2 / 4.0 - (9*y - 2)**2 / 4.0)
    t2 = 0.75 * np.exp(-(9*x + 1)**2 / 49.0 - (9*y + 1)**2 / 10.0)
    t3 = 0.50 * np.exp(-(9*x - 7)**2 / 4.0 - (9*y - 3)**2 / 4.0)
    t4 = 0.20 * np.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    return t1 + t2 + t3 - t4

def franke_torch(X: Tensor, noise_sd: float = 0.0) -> Tensor:
    x, y = X[..., 0], X[..., 1]
    t1 = 0.75 * torch.exp(-(9*x - 2)**2 / 4.0 - (9*y - 2)**2 / 4.0)
    t2 = 0.75 * torch.exp(-(9*x + 1)**2 / 49.0 - (9*y + 1)**2 / 10.0)
    t3 = 0.50 * torch.exp(-(9*x - 7)**2 / 4.0 - (9*y - 3)**2 / 4.0)
    t4 = 0.20 * torch.exp(-(9*x - 4)**2 - (9*y - 7)**2)
    f = t1 + t2 + t3 - t4
    if noise_sd > 0.0:
        f = f + noise_sd * torch.randn_like(f)
    return f.unsqueeze(-1)

def _rotate_anisowarp_torch(X: Tensor, center=(0.5, 0.5), deg: float = 35.0, scale=(1.0, 0.35)) -> Tensor:
    cx, cy = float(center[0]), float(center[1])
    th = math.radians(deg)
    R = torch.tensor([[math.cos(th), -math.sin(th)],
                      [math.sin(th),  math.cos(th)]], device=X.device, dtype=X.dtype)
    S = torch.tensor([[scale[0], 0.0],[0.0, scale[1]]], device=X.device, dtype=X.dtype)
    Xc = X - torch.tensor([cx, cy], device=X.device, dtype=X.dtype)
    return (Xc @ R.t()) @ S.t() + torch.tensor([cx, cy], device=X.device, dtype=X.dtype)

def franke_hard_torch(
    X: Tensor,
    noise_sd: float = 0.0,
    rotate_deg: float = 35.0,
    anisotropy: Tuple[float,float] = (1.0, 0.35),
    ripple_amp: float = 0.10,
    ripple_freq: Tuple[int,int] = (5,7),
    distractor_amp: float = 0.25,
    distractor_mu: Tuple[float,float] = (0.78, 0.82),
    distractor_sigma: float = 0.035,
    ridge_depth: float = 0.12,
    ridge_dir: Tuple[float,float] = (0.8, -0.6),
    ridge_sigma: float = 0.025,
    clip01: bool = True,
) -> Tensor:
    Xw = _rotate_anisowarp_torch(X, deg=rotate_deg, scale=anisotropy)
    base = franke_torch(Xw, noise_sd=0.0).squeeze(-1)
    kx, ky = ripple_freq
    rip = ripple_amp * torch.sin(2*math.pi*kx*X[...,0]) * torch.sin(2*math.pi*ky*X[...,1])
    mu = torch.tensor(distractor_mu, device=X.device, dtype=X.dtype)
    d = torch.exp(-0.5 * torch.sum(((X - mu)/distractor_sigma)**2, dim=-1))
    dist = distractor_amp * d
    v = torch.tensor(ridge_dir, device=X.device, dtype=X.dtype)
    v = v / torch.norm(v)
    proj = torch.sum((X - 0.5) * v, dim=-1)
    ridge = -ridge_depth * torch.exp(-0.5 * (proj / ridge_sigma)**2)
    f = base + rip + dist + ridge
    if clip01:
        f = torch.clamp(f, min=float(f.min().item()), max=float(f.max().item()))
    if noise_sd > 0.0:
        het = 0.5 + 0.5*torch.abs(torch.sin(2*math.pi*X[...,0]))*torch.abs(torch.sin(2*math.pi*X[...,1]))
        f = f + noise_sd * het * torch.randn_like(f)
    return f.unsqueeze(-1)

__all__ = ["franke_np", "franke_torch", "franke_hard_torch"]
