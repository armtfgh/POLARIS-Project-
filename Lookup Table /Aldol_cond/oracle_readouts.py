# oracle_readouts.py
from typing import Dict, Any, Tuple
import numpy as np
import torch
from franke import franke_torch, franke_hard_torch

def _argmax_on_grid(obj, grid_n: int = 201) -> Tuple[float, float]:
    xs = np.linspace(0, 1, grid_n)
    xx, yy = np.meshgrid(xs, xs)
    pts = np.c_[xx.ravel(), yy.ravel()]
    X = torch.tensor(pts, dtype=torch.get_default_dtype())
    with torch.no_grad():
        vals = obj(X).squeeze(-1).cpu().numpy()
    i = int(vals.argmax())
    return float(pts[i,0]), float(pts[i,1])

def perfect_readout_franke(hard: bool = False,
                           grid_n: int = 201,
                           dx: float = 0.08,
                           dy: float = 0.08,
                           bump_sigma: float = 0.10,
                           bump_amp: float = 0.9,
                           synergy_conf: float = 0.95) -> Dict[str, Any]:
    obj = franke_hard_torch if hard else franke_torch
    x1_star, x2_star = _argmax_on_grid(obj, grid_n=grid_n)

    def rng(lo, hi):  # clamp to [0,1]
        return [max(0.0, lo), min(1.0, hi)]

    return {
        "effects": {
            "x1": {"effect": "nonmonotone-peak", "scale": 1.0, "confidence": 0.99,
                   "range_hint": rng(x1_star - dx, x1_star + dx)},
            "x2": {"effect": "nonmonotone-peak", "scale": 1.0, "confidence": 0.99,
                   "range_hint": rng(x2_star - dy, x2_star + dy)},
        },
        "interactions": [
            {"pair": ["x1","x2"], "type": "synergy", "confidence": synergy_conf}
        ],
        "bumps": [
            {"mu": [x1_star, x2_star], "sigma": bump_sigma, "amp": bump_amp}
        ],
    }
