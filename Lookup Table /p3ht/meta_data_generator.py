"""
Meta-dataset generator for prior-learning experiments.

Produces a large corpus of synthetic response surfaces along with staged observation
sets and the corresponding "optimal" priors (effects/interactions/bumps) fitted
directly to each surface. The resulting JSON/NPZ files can be used to train
models or RAG pipelines that map sparse observations to priors.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np


# --------------------------------------------------------------------------------------
# Surface definitions
# --------------------------------------------------------------------------------------

def franke_surface(x1: np.ndarray, x2: np.ndarray, a: float = 0.75, b: float = 0.25) -> np.ndarray:
    term1 = a * np.exp(-(9 * x1 - 2) ** 2 / 4.0 - (9 * x2 - 2) ** 2 / 4.0)
    term2 = 0.75 * np.exp(-((9 * x1 + 1) ** 2) / 49.0 - (9 * x2 + 1) / 10.0)
    term3 = 0.5 * np.exp(-(9 * x1 - 7) ** 2 / 4.0 - (9 * x2 - 3) ** 2 / 4.0)
    term4 = -b * np.exp(-(9 * x1 - 4) ** 2 - (9 * x2 - 7) ** 2)
    return term1 + term2 + term3 + term4


def gaussian_mixture(x1: np.ndarray, x2: np.ndarray, centers: List[Tuple[float, float]], amps: List[float]) -> np.ndarray:
    out = np.zeros_like(x1)
    for (cx, cy), amp in zip(centers, amps):
        sx = random.uniform(0.05, 0.2)
        sy = random.uniform(0.05, 0.2)
        out += amp * np.exp(-(((x1 - cx) ** 2) / (2 * sx**2) + ((x2 - cy) ** 2) / (2 * sy**2)))
    return out


def polynomial_ridge(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return 1.2 * x1**2 - 0.8 * x2**2 + 0.5 * x1 * x2 - 0.3 * x1 + 0.2 * x2


def instantiate_surface(surface_type: str) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    if surface_type == "franke":
        a = random.uniform(0.6, 0.9)
        b = random.uniform(0.15, 0.35)
        return lambda x1, x2: franke_surface(x1, x2, a=a, b=b)
    if surface_type == "gauss_mix":
        centers = [(random.uniform(0.1, 0.9), random.uniform(0.1, 0.9)) for _ in range(random.randint(2, 4))]
        amps = [random.uniform(-0.3, 1.0) for _ in range(len(centers))]
        return lambda x1, x2: gaussian_mixture(x1, x2, centers=centers, amps=amps)
    if surface_type == "poly_ridge":
        c1, c2, c3 = random.uniform(0.8, 1.5), random.uniform(-1.2, -0.4), random.uniform(-0.5, 0.5)
        c4, c5 = random.uniform(-0.4, 0.4), random.uniform(-0.4, 0.4)
        return lambda x1, x2: c1 * x1**2 + c2 * x2**2 + c3 * x1 * x2 + c4 * x1 + c5 * x2
    return polynomial_ridge


# --------------------------------------------------------------------------------------
# Prior fitting utilities (replicating prior_gp schema)
# --------------------------------------------------------------------------------------

def evaluate_prior_surface(spec: Dict[str, Dict], X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    grid = np.stack([X1.ravel(), X2.ravel()], axis=-1)
    out = np.zeros(grid.shape[0], dtype=np.float64)
    d = 2

    def parse_dim(name: str) -> int:
        if name.startswith("x"):
            return int(name[1:]) - 1
        raise ValueError("Invalid dimension name")

    def sigmoid(z, center=0.5, k=6.0):
        return 1 / (1 + np.exp(-k * (z - center)))

    def gauss1d(z, mu=0.5, s=0.18):
        s = max(s, 1e-6)
        return np.exp(-0.5 * ((z - mu) / s) ** 2)

    for name, eff_spec in (spec.get("effects") or {}).items():
        idx = parse_dim(name)
        z = grid[:, idx]
        eff = eff_spec.get("effect", "flat").lower()
        scale = float(eff_spec.get("scale", 0.0))
        conf = float(eff_spec.get("confidence", 0.0))
        amp = 0.6 * scale * conf

        range_hint = eff_spec.get("range_hint")
        center = 0.5
        width = 0.18
        if isinstance(range_hint, (list, tuple)) and len(range_hint) == 2:
            lo, hi = float(range_hint[0]), float(range_hint[1])
            center = 0.5 * (lo + hi)
            width = max(abs(hi - lo) / 3.0, 0.05)

        if eff == "increase":
            out += amp * sigmoid(z, center=center)
        elif eff == "decrease":
            out -= amp * sigmoid(z, center=center)
        elif eff in {"peak", "nonmonotone-peak"}:
            out += amp * gauss1d(z, mu=center, s=width)
        elif eff in {"valley", "nonmonotone-valley"}:
            out -= amp * gauss1d(z, mu=center, s=width)

    for inter in (spec.get("interactions") or []):
        idx_a = parse_dim(inter["vars"][0])
        idx_b = parse_dim(inter["vars"][1])
        itype = inter.get("type", "synergy")
        sign = 1.0 if itype == "synergy" else -1.0
        scale = float(inter.get("scale", 0.0))
        conf = float(inter.get("confidence", 0.0))
        amp = 0.2 * max(scale, 0.0) * conf
        term = grid[:, idx_a] * grid[:, idx_b]
        out += sign * amp * term

    for bump in (spec.get("bumps") or []):
        mu = np.array(bump.get("mu", [0.5, 0.5]), dtype=np.float64)
        sigma = np.array(bump.get("sigma", [0.15, 0.15]), dtype=np.float64)
        sigma = np.clip(sigma, 1e-6, None)
        amp = float(bump.get("amp", 0.1))
        diff = (grid - mu) / sigma
        gauss = np.exp(-0.5 * np.sum(diff**2, axis=1))
        out += amp * gauss

    return out.reshape(X1.shape)


def fit_prior_to_surface(X1: np.ndarray, X2: np.ndarray, surface: np.ndarray) -> Dict[str, Dict]:
    effects = {}
    mean_x1 = surface.mean(axis=0)
    mean_x2 = surface.mean(axis=1)

    if np.var(mean_x1) > 1e-4:
        idx = np.argmax(mean_x1)
        eff_type = "peak" if random.random() < 0.5 else "increase"
        effects["x1"] = {
            "effect": eff_type,
            "scale": float(np.clip((mean_x1.max() - mean_x1.min()) / (surface.std() + 1e-6), 0.5, 1.2)),
            "confidence": 0.9,
            "range_hint": [max(0.0, X1[0, idx] - 0.2), min(1.0, X1[0, idx] + 0.2)],
        }

    if np.var(mean_x2) > 1e-4:
        idx = np.argmin(mean_x2)
        eff_type = "valley" if random.random() < 0.5 else "decrease"
        effects["x2"] = {
            "effect": eff_type,
            "scale": float(np.clip((mean_x2.max() - mean_x2.min()) / (surface.std() + 1e-6), 0.5, 1.2)),
            "confidence": 0.9,
            "range_hint": [max(0.0, X2[idx, 0] - 0.2), min(1.0, X2[idx, 0] + 0.2)],
        }

    residual = surface - evaluate_prior_surface({"effects": effects, "interactions": [], "bumps": []}, X1, X2)
    U, S, Vt = np.linalg.svd(residual, full_matrices=False)
    inter = {
        "vars": ["x1", "x2"],
        "type": "synergy" if S[0] >= 0 else "antagonism",
        "scale": float(np.clip(abs(S[0]) / (surface.std() + 1e-6), 0.4, 1.5)),
        "confidence": 0.9,
    }
    residual -= evaluate_prior_surface({"effects": {}, "interactions": [inter], "bumps": []}, X1, X2)

    flat = residual.ravel()
    idx_top = np.argsort(flat)[::-1][:2]
    bumps = []
    for idx in idx_top:
        i, j = np.unravel_index(idx, surface.shape)
        amp = float(residual[i, j])
        if abs(amp) < 0.02:
            continue
        bumps.append(
            {
                "mu": [float(X1[0, j]), float(X2[i, 0])],
                "sigma": [0.08, 0.08],
                "amp": amp,
            }
        )

    return {"effects": effects, "interactions": [inter], "bumps": bumps}


def infer_effect_from_samples(points: np.ndarray, values: np.ndarray, dim: int) -> Dict[str, float] | None:
    if len(values) < 3:
        return None
    coords = points[:, dim]
    corr = np.corrcoef(coords, values)[0, 1] if len(values) > 2 else 0.0
    corr = float(np.nan_to_num(corr))
    conf = float(np.clip(0.35 + 0.03 * len(values), 0.35, 0.9))
    range_hint = [float(np.percentile(coords, 25)), float(np.percentile(coords, 75))]
    if abs(corr) >= 0.35:
        effect = "increase" if corr > 0 else "decrease"
        scale = float(np.clip(abs(corr) + 0.3, 0.4, 1.1))
        return {"effect": effect, "scale": scale, "confidence": conf, "range_hint": range_hint}
    low = values[coords < range_hint[0]]
    high = values[coords > range_hint[1]]
    mid_mask = (coords >= range_hint[0]) & (coords <= range_hint[1])
    mid = values[mid_mask] if mid_mask.any() else values
    mean_mid = float(mid.mean()) if len(mid) else float(values.mean())
    mean_edges = float(np.concatenate([low, high]).mean()) if len(low) + len(high) > 0 else float(values.mean())
    if mean_mid > mean_edges + 0.05:
        effect = "peak"
    elif mean_mid + 0.05 < mean_edges:
        effect = "valley"
    else:
        effect = "increase" if corr >= 0 else "decrease"
    return {"effect": effect, "scale": 0.55, "confidence": conf * 0.8, "range_hint": range_hint}


def infer_interaction_from_samples(points: np.ndarray, values: np.ndarray) -> Dict[str, float] | None:
    if len(values) < 4:
        return None
    prod = (points[:, 0] - points[:, 0].mean()) * (points[:, 1] - points[:, 1].mean())
    corr = np.corrcoef(prod, values)[0, 1]
    corr = float(np.nan_to_num(corr))
    if abs(corr) < 0.2:
        return None
    return {
        "vars": ["x1", "x2"],
        "type": "synergy" if corr > 0 else "antagonism",
        "scale": float(np.clip(abs(corr) * 1.5, 0.2, 1.2)),
        "confidence": float(np.clip(0.3 + 0.02 * len(values), 0.3, 0.9)),
    }


def infer_bumps_from_samples(points: np.ndarray, values: np.ndarray, max_bumps: int = 2) -> List[Dict[str, float]]:
    if len(values) == 0:
        return []
    idx = np.argsort(values)[::-1][:max_bumps]
    mean_val = float(values.mean())
    bumps = []
    for i in idx:
        amp = float(values[i] - mean_val)
        if abs(amp) < 0.05:
            continue
        bumps.append(
            {
                "mu": [float(points[i, 0]), float(points[i, 1])],
                "sigma": [0.12, 0.12],
                "amp": amp,
            }
        )
    return bumps


def infer_prior_from_samples(points: np.ndarray, values: np.ndarray) -> Dict[str, Dict]:
    pts = np.asarray(points, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64)
    prior = {"effects": {}, "interactions": [], "bumps": []}
    for dim, name in enumerate(["x1", "x2"]):
        eff = infer_effect_from_samples(pts, vals, dim)
        if eff:
            prior["effects"][name] = eff
    inter = infer_interaction_from_samples(pts, vals)
    if inter:
        prior["interactions"].append(inter)
    prior["bumps"] = infer_bumps_from_samples(pts, vals)
    return prior


# --------------------------------------------------------------------------------------
# Dataset generation
# --------------------------------------------------------------------------------------

@dataclass
class StageRecord:
    surface_id: int
    surface_type: str
    stage_idx: int
    n_samples: int
    sample_points: List[List[float]]
    sample_values: List[float]
    prior: Dict[str, Dict]
    true_prior: Dict[str, Dict]


def generate_dataset(
    out_dir: Path,
    n_surfaces: int,
    observation_sizes: List[int],
    grid_size: int = 160,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: List[StageRecord] = []

    xs = np.linspace(0.0, 1.0, grid_size)
    X1, X2 = np.meshgrid(xs, xs)

    for sid in range(n_surfaces):
        surface_type = random.choice(["franke", "gauss_mix", "poly_ridge"])
        surface_fn = instantiate_surface(surface_type)
        surface = surface_fn(X1, X2)
        true_prior = fit_prior_to_surface(X1, X2, surface)
        for stage_idx, n_samples in enumerate(observation_sizes, start=1):
            pts = rng.random((n_samples, 2))
            vals = surface_fn(pts[:, 0], pts[:, 1])
            stage_prior = infer_prior_from_samples(pts, vals)
            records.append(
                StageRecord(
                    surface_id=sid,
                    surface_type=surface_type,
                    stage_idx=stage_idx,
                    n_samples=n_samples,
                    sample_points=pts.tolist(),
                    sample_values=vals.tolist(),
                    prior=stage_prior,
                    true_prior=true_prior,
                )
            )

    with open(out_dir / "meta_dataset.json", "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in records], f)
    np.savez(out_dir / "surfaces.npz", X1=X1, X2=X2)
    print(f"[meta_data_generator] Saved {len(records)} records to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic meta dataset for prior learning.")
    parser.add_argument("--out-dir", type=Path, default=Path("meta_dataset"), help="Output directory.")
    parser.add_argument("--n-surfaces", type=int, default=200, help="Number of synthetic surfaces.")
    parser.add_argument("--observation-sizes", type=str, default="5,10,20", help="Comma-separated sample sizes.")
    parser.add_argument("--grid-size", type=int, default=160, help="Resolution of the dense grid.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    obs_sizes = [int(s) for s in args.observation_sizes.split(",")]
    generate_dataset(
        out_dir=args.out_dir,
        n_surfaces=args.n_surfaces,
        observation_sizes=obs_sizes,
        grid_size=args.grid_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
