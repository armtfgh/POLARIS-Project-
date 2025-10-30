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
        z1, z2 = X[..., 0], X[..., 1]
        out = torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)

        def _sigmoid(z, k: float = 6.0): return 1.0 / (1.0 + torch.exp(-k * (z - 0.5)))
        def _gauss1d(z, mu: float, s: float): return torch.exp(-0.5 * ((z - mu) / (s + 1e-12)) ** 2)

        for name, spec in self.effects.items():
            eff = spec.get("effect", "flat")
            sc = float(spec.get("scale", 0.0)); conf = float(spec.get("confidence", 0.0))
            rh = spec.get("range_hint", None); amp = 0.6 * sc * conf
            if amp == 0.0: continue
            mu = 0.5
            if isinstance(rh, (list, tuple)) and len(rh) == 2:
                mu = 0.5 * (float(rh[0]) + float(rh[1]))
            z = z1 if name == "x1" else z2
            if eff == "increase": out = out + amp * _sigmoid(z)
            elif eff == "decrease": out = out - amp * _sigmoid(z)
            elif eff == "nonmonotone-peak": out = out + amp * _gauss1d(z, mu=mu, s=0.18)
            elif eff == "nonmonotone-valley": out = out - amp * _gauss1d(z, mu=mu, s=0.18)

        for it in self.interactions:
            pair = it.get("pair", [])
            if pair == ["x1", "x2"] or pair == ["x2", "x1"]:
                sign = 1.0 if it.get("type", "synergy") == "synergy" else -1.0
                conf = float(it.get("confidence", 0.0))
                out = out + 0.2 * sign * conf * (z1 * z2)

        for b in self.bumps:
            mu = torch.tensor(b.get("mu", [0.5, 0.5]), device=X.device, dtype=X.dtype)
            sig = float(b.get("sigma", 0.15)); amp = float(b.get("amp", 0.1))
            out = out + amp * torch.exp(-0.5 * torch.sum(((X - mu) / sig) ** 2, dim=-1))
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
