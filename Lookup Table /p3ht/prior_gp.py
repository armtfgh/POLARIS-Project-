from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import torch
from torch import Tensor
from botorch.models import SingleTaskGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from botorch.fit import fit_gpytorch_mll
import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

@dataclass
class Prior:
    effects: Dict[str, Dict[str, Any]]
    interactions: List[Dict[str, Any]]
    bumps: List[Dict[str, Any]]

    def m0_torch(self, X: Tensor) -> Tensor:
        """Evaluate the prior mean m0(X) for normalized inputs X in [0,1]^d."""
        if X.ndim == 1:
            X = X.unsqueeze(0)
        d = X.shape[-1]
        out = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

        def _parse_dim(name: Any) -> Optional[int]:
            if isinstance(name, int):
                idx = name
            elif isinstance(name, str) and name.startswith("x"):
                try:
                    idx = int(name[1:]) - 1
                except ValueError:
                    return None
            else:
                return None
            return idx if 0 <= idx < d else None

        def _sigmoid(z: Tensor, center: float = 0.5, k: float = 6.0) -> Tensor:
            return torch.sigmoid(k * (z - center))

        def _gauss1d(z: Tensor, mu: float, s: float) -> Tensor:
            s = max(s, 1e-6)
            return torch.exp(-0.5 * ((z - mu) / s) ** 2)

        # ----- main effects -----
        for name, spec in (self.effects or {}).items():
            idx = _parse_dim(name)
            if idx is None:
                continue
            z = X[..., idx]
            eff = str(spec.get("effect", "flat")).lower()
            scale = float(spec.get("scale", 0.0))
            conf = float(spec.get("confidence", 0.0))
            amp = 0.6 * scale * conf
            if amp == 0.0:
                continue

            range_hint = spec.get("range_hint")
            center = 0.5
            width = 0.18
            if isinstance(range_hint, (list, tuple)) and len(range_hint) == 2:
                lo, hi = float(range_hint[0]), float(range_hint[1])
                center = 0.5 * (lo + hi)
                width = max(abs(hi - lo) / 3.0, 0.05)

            if eff in {"increase", "increasing"}:
                out = out + amp * _sigmoid(z, center=center)
            elif eff in {"decrease", "decreasing"}:
                out = out - amp * _sigmoid(z, center=center)
            elif eff in {"nonmonotone-peak", "peak"}:
                out = out + amp * _gauss1d(z, mu=center, s=width)
            elif eff in {"nonmonotone-valley", "valley"}:
                out = out - amp * _gauss1d(z, mu=center, s=width)
            # "flat" or unknown => no contribution

        # ----- pairwise interactions -----
        for inter in (self.interactions or []):
            pair = inter.get("vars") or inter.get("pair") or inter.get("indices")
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            idx_a = _parse_dim(pair[0])
            idx_b = _parse_dim(pair[1])
            if idx_a is None or idx_b is None:
                continue

            itype = str(inter.get("type", "synergy")).lower()
            sign = 1.0
            if itype in {"antagonism", "tradeoff", "negative"}:
                sign = -1.0
            strength = max(float(inter.get("scale", inter.get("strength", 0.0))), 0.0)
            conf = max(float(inter.get("confidence", 0.0)), 0.0)
            if strength == 0.0 and conf == 0.0:
                conf = 0.5
            amp = 0.2 * (strength if strength > 0 else 1.0) * conf
            term = (X[..., idx_a] * X[..., idx_b])
            out = out + sign * amp * term

        # ----- bumps -----
        for bump in (self.bumps or []):
            if not bump:
                continue
            mu_vals = bump.get("mu")
            if mu_vals is None:
                continue
            mu = torch.tensor(list(mu_vals)[:d], device=X.device, dtype=X.dtype)
            if mu.numel() == 0:
                mu = torch.full((d,), 0.5, device=X.device, dtype=X.dtype)
            elif mu.numel() < d:
                mu = torch.cat([mu, torch.full((d - mu.numel(),), 0.5, device=X.device, dtype=X.dtype)], dim=0)

            sigma_vals = bump.get("sigma", 0.15)
            if isinstance(sigma_vals, (list, tuple)):
                sigma = torch.tensor(list(sigma_vals)[:d], device=X.device, dtype=X.dtype)
                if sigma.numel() == 0:
                    sigma = torch.full((d,), 0.15, device=X.device, dtype=X.dtype)
                elif sigma.numel() < d:
                    sigma = torch.cat([sigma, torch.full((d - sigma.numel(),), sigma[-1], device=X.device, dtype=X.dtype)], dim=0)
            else:
                sigma = torch.full((d,), float(sigma_vals), device=X.device, dtype=X.dtype)
            sigma = torch.clamp(sigma, min=1e-6)

            amp = float(bump.get("amp", 0.1))
            diff = (X - mu) / sigma
            gauss = torch.exp(-0.5 * torch.sum(diff ** 2, dim=-1))
            out = out + amp * gauss

        return out

from botorch.models.model import Model
from botorch.models import SingleTaskGP
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from torch import Tensor
from typing import Optional, List

class GPWithPriorMean(Model):
    def __init__(self, base_gp: SingleTaskGP, prior: Prior, m0_scale: float = 1.0):
        super().__init__() # <-- important!
        self.base_gp = base_gp # safe to assign after super().__init__
        self.prior = prior
        self.m0_scale = float(m0_scale)

    @property
    def num_outputs(self) -> int:
        return 1

    def posterior(self, X: Tensor, observation_noise: bool = False, **kwargs) -> GPyTorchPosterior:
        base_post = self.base_gp.posterior(X, observation_noise=observation_noise, **kwargs)
        mvn = base_post.mvn
        m0 = self.prior.m0_torch(X).reshape(mvn.mean.shape)
        new_mvn = MultivariateNormal(mean=mvn.mean + self.m0_scale * m0,
        covariance_matrix=mvn.covariance_matrix)
        return GPyTorchPosterior(new_mvn)

    def condition_on_observations(self, X: Tensor, Y: Tensor, noise: Optional[Tensor] = None, **kwargs):
        cm = self.base_gp.condition_on_observations(X=X, Y=Y, noise=noise, **kwargs)
        return GPWithPriorMean(base_gp=cm, prior=self.prior, m0_scale=self.m0_scale)

    def fantasize(self, X: Tensor, sampler, observation_noise: bool = True, **kwargs):
        fm = self.base_gp.fantasize(X=X, sampler=sampler, observation_noise=observation_noise, **kwargs)
        return GPWithPriorMean(base_gp=fm, prior=self.prior, m0_scale=self.m0_scale)

    def subset_output(self, idcs: List[int]):
        return self

def fit_residual_gp(X: Tensor, Y: Tensor, prior: Prior) -> Tuple[SingleTaskGP, float]:
    m0 = prior.m0_torch(X).reshape(-1)
    yv = Y.reshape(-1)
    m0c = m0 - m0.mean(); yc = yv - yv.mean()
    denom = torch.dot(m0c, m0c).item()
    alpha = (torch.dot(m0c, yc).item() / (denom + 1e-12)) if denom > 0 else 0.0
    Y_resid = Y - alpha * m0.unsqueeze(-1)
    gp = SingleTaskGP(X, Y_resid).to(DEVICE)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(DEVICE)
    fit_gpytorch_mll(mll)
    return gp, alpha

def alignment_on_obs(X: Tensor, Y: Tensor, prior: Prior) -> float:
    m0 = prior.m0_torch(X).reshape(-1); yv = Y.reshape(-1)
    m0c = m0 - m0.mean(); yc = yv - yv.mean()
    num = torch.dot(m0c, yc).item()
    den = torch.sqrt(torch.dot(m0c, m0c) * torch.dot(yc, yc) + 1e-12).item()
    return num / den if den > 0 else 0.0

__all__ = ["Prior", "GPWithPriorMean", "fit_residual_gp", "alignment_on_obs"]
