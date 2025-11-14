#!/usr/bin/env python3

#%%
"""
LLM-driven prior benchmarking on the Franke surface.

Workflow
--------
1. Start with zero knowledge of the Franke surface.
2. Sample the surface in batches (default 5 points per batch) to build a sparse map.
3. After each batch, prompt an LLM (or a heuristic fallback) to propose a prior map
   using the same schema as `prior_gp.Prior`.
4. Evaluate alignment between the proposed prior and the true Franke surface.
5. Iterate, letting the LLM accumulate context (observations + past alignments) to
   refine the prior map. Track how many samples are needed to reach a high alignment.

Run:
    python llm_prior_benchmark_franke.py --iterations 8 --batch-size 5

Set OPENAI_API_KEY in the environment to enable real LLM calls. Otherwise the script
falls back to a deterministic heuristic prior generator so the pipeline still runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import numpy as np
import httpx
import matplotlib.pyplot as plt

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


# ---------------------------------------------------------------------------
# Franke surface utilities
# ---------------------------------------------------------------------------

def franke_surface(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    term1 = 0.75 * np.exp(-(9 * x1 - 2) ** 2 / 4.0 - (9 * x2 - 2) ** 2 / 4.0)
    term2 = 0.75 * np.exp(-((9 * x1 + 1) ** 2) / 49.0 - (9 * x2 + 1) / 10.0)
    term3 = 0.5 * np.exp(-(9 * x1 - 7) ** 2 / 4.0 - (9 * x2 - 3) ** 2 / 4.0)
    term4 = -0.2 * np.exp(-(9 * x1 - 4) ** 2 - (9 * x2 - 7) ** 2)
    return term1 + term2 + term3 + term4


def sample_franke(batch_size: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    pts = rng.random((batch_size, 2), dtype=np.float64)
    vals = franke_surface(pts[:, 0], pts[:, 1])
    return pts, vals


def make_grid(n: int = 220) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(0.0, 1.0, n)
    X1, X2 = np.meshgrid(xs, xs)
    truth = franke_surface(X1, X2)
    return X1, X2, truth


# ---------------------------------------------------------------------------
# Prior evaluation (NumPy replica of prior_gp.Prior)
# ---------------------------------------------------------------------------

def evaluate_prior_surface(spec: Dict[str, Any], X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    grid = np.stack([X1.ravel(), X2.ravel()], axis=-1)
    d = grid.shape[1]
    out = np.zeros(grid.shape[0], dtype=np.float64)

    def parse_dim(name: Any) -> Optional[int]:
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

    def sigmoid(z: np.ndarray, center: float = 0.5, k: float = 6.0) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-k * (z - center)))

    def gauss1d(z: np.ndarray, mu: float, s: float) -> np.ndarray:
        s = max(s, 1e-6)
        return np.exp(-0.5 * ((z - mu) / s) ** 2)

    for name, eff_spec in (spec.get("effects") or {}).items():
        idx = parse_dim(name)
        if idx is None:
            continue
        z = grid[:, idx]
        eff = str(eff_spec.get("effect", "flat")).lower()
        scale = float(eff_spec.get("scale", 0.0))
        conf = float(eff_spec.get("confidence", 0.0))
        amp = 0.6 * scale * conf
        if amp == 0.0:
            continue

        range_hint = eff_spec.get("range_hint")
        center = 0.5
        width = 0.18
        if isinstance(range_hint, (list, tuple)) and len(range_hint) == 2:
            lo, hi = float(range_hint[0]), float(range_hint[1])
            center = 0.5 * (lo + hi)
            width = max(abs(hi - lo) / 3.0, 0.05)

        if eff in {"increase", "increasing"}:
            out = out + amp * sigmoid(z, center=center)
        elif eff in {"decrease", "decreasing"}:
            out = out - amp * sigmoid(z, center=center)
        elif eff in {"peak", "nonmonotone-peak"}:
            out = out + amp * gauss1d(z, mu=center, s=width)
        elif eff in {"valley", "nonmonotone-valley"}:
            out = out - amp * gauss1d(z, mu=center, s=width)

    for inter in (spec.get("interactions") or []):
        pair = inter.get("vars") or inter.get("pair") or inter.get("indices")
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        idx_a = parse_dim(pair[0])
        idx_b = parse_dim(pair[1])
        if idx_a is None or idx_b is None:
            continue

        itype = str(inter.get("type", "synergy")).lower()
        sign = -1.0 if itype in {"antagonism", "tradeoff", "negative"} else 1.0
        strength = max(float(inter.get("scale", inter.get("strength", 0.0))), 0.0)
        conf = max(float(inter.get("confidence", 0.0)), 0.0)
        if strength == 0.0 and conf == 0.0:
            conf = 0.5
        amp = 0.2 * (strength if strength > 0 else 1.0) * conf
        term = grid[:, idx_a] * grid[:, idx_b]
        out = out + sign * amp * term

    for bump in (spec.get("bumps") or []):
        if not bump:
            continue
        mu_vals = bump.get("mu")
        if mu_vals is None:
            continue
        mu = np.array(list(mu_vals)[:d], dtype=np.float64)
        if mu.size == 0:
            mu = np.full(d, 0.5, dtype=np.float64)
        elif mu.size < d:
            mu = np.concatenate([mu, np.full(d - mu.size, 0.5, dtype=np.float64)])

        sigma_vals = bump.get("sigma", 0.15)
        if isinstance(sigma_vals, (list, tuple)):
            sigma = np.array(list(sigma_vals)[:d], dtype=np.float64)
            if sigma.size == 0:
                sigma = np.full(d, 0.15, dtype=np.float64)
            elif sigma.size < d:
                sigma = np.concatenate([sigma, np.full(d - sigma.size, sigma[-1], dtype=np.float64)])
        else:
            sigma = np.full(d, float(sigma_vals), dtype=np.float64)
        sigma = np.clip(sigma, 1e-6, None)

        amp = float(bump.get("amp", 0.1))
        diff = (grid - mu) / sigma
        gauss = np.exp(-0.5 * np.sum(diff ** 2, axis=1))
        out = out + amp * gauss

    return out.reshape(X1.shape)


def alignment_metrics(truth: np.ndarray, prior: np.ndarray) -> Dict[str, float]:
    tf = truth.ravel()
    pf = prior.ravel()
    rho = float(np.corrcoef(tf, pf)[0, 1])
    rmse = float(np.sqrt(np.mean((tf - pf) ** 2)))
    denom = np.dot(pf, pf) + 1e-12
    alpha = float(np.dot(tf, pf) / denom)
    return {"rho": rho, "rmse": rmse, "alpha": alpha}


def describe_readout(readout: Dict[str, Any]) -> str:
    """Pretty JSON dump for console display."""
    try:
        return json.dumps(readout, indent=2, sort_keys=True)
    except TypeError:
        return str(readout)


def render_iteration_figure(
    iteration: int,
    X1: np.ndarray,
    X2: np.ndarray,
    truth: np.ndarray,
    prior: np.ndarray,
    X_obs: np.ndarray,
    y_obs: np.ndarray,
    save_path: Optional[Path],
    show: bool,
) -> None:
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    levels = np.linspace(truth.min(), truth.max(), 30)
    cm = "viridis"
    scatter_kwargs = {}
    if len(X_obs) > 0:
        scatter_kwargs = {
            "c": y_obs,
            "cmap": "coolwarm",
            "edgecolor": "k",
            "linewidths": 0.4,
            "s": 30,
        }

    im0 = axes[0].contourf(X1, X2, truth, levels=levels, cmap=cm)
    if scatter_kwargs:
        axes[0].scatter(X_obs[:, 0], X_obs[:, 1], **scatter_kwargs)
    axes[0].set_title("Ground truth contour")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")

    im1 = axes[1].contourf(X1, X2, prior, levels=levels, cmap=cm)
    if scatter_kwargs:
        axes[1].scatter(X_obs[:, 0], X_obs[:, 1], **scatter_kwargs)
    axes[1].set_title("Prior mean contour")
    axes[1].set_xlabel("$x_1$")
    axes[1].set_ylabel("$x_2$")

    resid = truth - prior
    res_levels = np.linspace(-np.max(np.abs(resid)), np.max(np.abs(resid)), 30)
    im2 = axes[2].contourf(X1, X2, resid, levels=res_levels, cmap="coolwarm")
    axes[2].set_title("Residual (truth - prior)")
    axes[2].set_xlabel("$x_1$")
    axes[2].set_ylabel("$x_2$")

    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="Value")
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="Value")
    fig.colorbar(im2, ax=axes[2], shrink=0.8, label="Residual")
    fig.suptitle(f"Iteration {iteration} prior vs Franke surface")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        fig.savefig(save_path, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Observation summaries passed to the LLM
# ---------------------------------------------------------------------------

def topk_summary(X: np.ndarray, y: np.ndarray, k: int = 3) -> str:
    idx_sorted = np.argsort(y)[::-1]
    parts = []
    for rank in range(min(k, len(idx_sorted))):
        idx = idx_sorted[rank]
        parts.append(
            f"Top {rank+1}: y={y[idx]:.3f} at (x1={X[idx,0]:.3f}, x2={X[idx,1]:.3f})"
        )
    idx_worst = np.argsort(y)[:min(2, len(y))]
    for i, idx in enumerate(idx_worst, 1):
        parts.append(
            f"Bottom {i}: y={y[idx]:.3f} at (x1={X[idx,0]:.3f}, x2={X[idx,1]:.3f})"
        )
    return "\n".join(parts)


def coarse_bin_summary(X: np.ndarray, y: np.ndarray, bins: int = 4) -> str:
    if len(X) == 0:
        return "No data yet."
    H, xedges, yedges = np.histogram2d(X[:, 0], X[:, 1], bins=bins, weights=y)
    counts, _, _ = np.histogram2d(X[:, 0], X[:, 1], bins=bins)
    avg = np.divide(H, counts, out=np.full_like(H, np.nan), where=counts > 0)
    cells = []
    for i in range(bins):
        for j in range(bins):
            if counts[i, j] == 0:
                continue
            cells.append(
                {
                    "x_range": (float(xedges[i]), float(xedges[i + 1])),
                    "y_range": (float(yedges[j]), float(yedges[j + 1])),
                    "mean_y": float(avg[i, j]),
                    "n": int(counts[i, j]),
                }
            )
    cells.sort(key=lambda c: c["mean_y"], reverse=True)
    lines = []
    for cell in cells[:6]:
        lines.append(
            f"Bin x∈[{cell['x_range'][0]:.2f},{cell['x_range'][1]:.2f}], "
            f"y∈[{cell['y_range'][0]:.2f},{cell['y_range'][1]:.2f}] "
            f"mean={cell['mean_y']:.3f} (n={cell['n']})"
        )
    if not lines:
        return "Bins exist but no values recorded."
    return "\n".join(lines)


def history_summary(logs: List[Dict[str, Any]]) -> str:
    if not logs:
        return "No prior proposals yet."
    lines = []
    for log in logs:
        lines.append(
            f"Iter {log['iteration']}: rho={log['rho']:.3f}, alpha={log['alpha']:.3f}, prior={log.get('readout_text','?')}"
        )
    return "\n".join(lines)


def hotspot_summary(X: np.ndarray, y: np.ndarray, top_k: int = 3) -> str:
    if len(X) == 0:
        return "No hotspots detected yet."
    idx = np.argsort(y)[-min(top_k, len(y)):]
    lines = []
    for i, idc in enumerate(idx[::-1], 1):
        center = X[idc]
        amp = y[idc]
        radius = 0.08
        lines.append(
            f"Hotspot {i}: center≈({center[0]:.2f}, {center[1]:.2f}), amp≈{amp:.3f}, suggested sigma≈[{radius:.2f}, {radius:.2f}]"
        )
    return "\n".join(lines)


def directional_summary(X: np.ndarray, y: np.ndarray) -> str:
    if len(X) == 0:
        return "No directional evidence yet."
    segments = {
        "x1<0.3": y[X[:, 0] < 0.3],
        "x1>0.7": y[X[:, 0] > 0.7],
        "x2<0.3": y[X[:, 1] < 0.3],
        "x2>0.7": y[X[:, 1] > 0.7],
    }
    lines = []
    for label, vals in segments.items():
        if len(vals) == 0:
            continue
        lines.append(f"{label} mean={np.mean(vals):.3f} (n={len(vals)})")
    if not lines:
        return "Directional bins empty."
    return "\n".join(lines)


def summarize_readout(readout: Dict[str, Any]) -> str:
    parts = []
    effects = readout.get("effects", {}) or {}
    for dim in ("x1", "x2"):
        if dim in effects:
            eff = effects[dim]
            parts.append(
                f"{dim}: {eff.get('effect','?')} (scale={eff.get('scale','?')}, conf={eff.get('confidence','?')}, "
                f"range={eff.get('range_hint','?')})"
            )
    interactions = readout.get("interactions") or []
    if interactions:
        it = interactions[0]
        parts.append(
            f"interaction: {it.get('type','?')} scale={it.get('scale','?')} conf={it.get('confidence','?')}"
        )
    bumps = readout.get("bumps") or []
    if bumps:
        b = bumps[0]
        parts.append(f"bump at {b.get('mu','?')} amp={b.get('amp','?')}")
    return "; ".join(parts) if parts else "No prior description."


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You help design human-readable priors for Bayesian optimization on a 2-D Franke-like surface.
You receive sparse observations. You must output a JSON prior in the format:
{
  "effects": {
    "x1": {"effect": "increase|decrease|peak|valley", "scale": float, "confidence": float, "range_hint": [low, high]},
    "x2": {...}
  },
  "interactions": [
    {"vars": ["x1","x2"], "type": "synergy|antagonism", "scale": float, "confidence": float}
  ],
  "bumps": [
    {"mu": [x1_center, x2_center], "sigma": [sx, sy], "amp": float}
  ]
}
Only include fields you are confident about. Keep JSON valid with double quotes. No commentary outside the JSON."""


PROMPT_TEMPLATE = """We have observed {n_samples} samples on the Franke surface.

Current descriptive statistics:
    • Mean y = {mean_y:.3f}, Std y = {std_y:.3f}
    • Best y = {best_y:.3f} at (x1={best_x1:.3f}, x2={best_x2:.3f})
    • Worst y = {worst_y:.3f} at (x1={worst_x1:.3f}, x2={worst_x2:.3f})

Top observations:
{topk}

Directional aggregates:
{directional}

Coarse spatial averages:
{heatmap}

Previous priors:
{history}
Most recent prior summary:
{last_readout}

Hotspot suggestions (candidates for bumps):
{hotspots}

Please propose an updated prior JSON (effects/interactions/bumps) informed by these samples.
You may include multiple bumps if there appear to be several hotspots, and broader bumps are acceptable when evidence is diffuse.
Favor smooth, simple shapes unless evidence suggests otherwise."""


class LLMPriorAgent:
    def __init__(
        self,
        enabled: bool,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_retries: int = 3,
    ) -> None:
        self.enabled = enabled and OpenAI is not None and bool(os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.history: List[Dict[str, str]] = []
        self.client: Optional[OpenAI] = None
        if self.enabled and OpenAI is not None:
            self.client = OpenAI(http_client=httpx.Client(verify=False))

    def propose(self, prompt: str) -> Dict[str, Any]:
        if not self.client:
            return self._heuristic_prior()

        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + self.history + [{"role": "user", "content": prompt}]
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=messages,
                )
                text = response.choices[0].message.content or ""
                self.history.extend(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": text},
                    ]
                )
                parsed = self._extract_json(text)
                if parsed:
                    return parsed
            except Exception as exc:  # pragma: no cover
                print(f"[LLMPriorAgent] Attempt {attempt} failed: {exc}")
        print("[LLMPriorAgent] Falling back to heuristic prior after LLM failures.")
        return self._heuristic_prior()

    def _heuristic_prior(self) -> Dict[str, Any]:
        """Simple fallback prior increasing x1, peaking x2."""
        return {
            "effects": {
                "x1": {"effect": "increase", "scale": 0.8, "confidence": 0.7, "range_hint": [0.2, 0.9]},
                "x2": {"effect": "peak", "scale": 0.6, "confidence": 0.6, "range_hint": [0.2, 0.5]},
            },
            "interactions": [
                {"vars": ["x1", "x2"], "type": "synergy", "scale": 0.5, "confidence": 0.6}
            ],
            "bumps": [
                {"mu": [0.7, 0.25], "sigma": [0.08, 0.06], "amp": 0.4},
                {"mu": [0.3, 0.7], "sigma": [0.07, 0.07], "amp": -0.3},
            ],
        }

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        try:
            match = re.search(r"\{.*\}", text, re.S)
            if not match:
                return None
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    iterations: int,
    batch_size: int,
    seed: int,
    use_llm: bool,
    model: str,
    temperature: float,
    fig_dir: Optional[Path] = None,
    show_figs: bool = True,
) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    X_obs = np.zeros((0, 2), dtype=np.float64)
    y_obs = np.zeros((0,), dtype=np.float64)
    logs: List[Dict[str, Any]] = []

    X1, X2, truth = make_grid()
    agent = LLMPriorAgent(enabled=use_llm, model=model, temperature=temperature)
    fig_dir = Path(fig_dir) if fig_dir else None
    if fig_dir:
        fig_dir.mkdir(parents=True, exist_ok=True)

    for iteration in range(1, iterations + 1):
        X_new, y_new = sample_franke(batch_size, rng)
        X_obs = np.vstack([X_obs, X_new])
        y_obs = np.concatenate([y_obs, y_new])

        prompt = PROMPT_TEMPLATE.format(
            n_samples=len(y_obs),
            mean_y=float(np.mean(y_obs)),
            std_y=float(np.std(y_obs)),
            best_y=float(np.max(y_obs)),
            best_x1=float(X_obs[np.argmax(y_obs), 0]),
            best_x2=float(X_obs[np.argmax(y_obs), 1]),
            worst_y=float(np.min(y_obs)),
            worst_x1=float(X_obs[np.argmin(y_obs), 0]),
            worst_x2=float(X_obs[np.argmin(y_obs), 1]),
            topk=topk_summary(X_obs, y_obs),
            directional=directional_summary(X_obs, y_obs),
            heatmap=coarse_bin_summary(X_obs, y_obs),
            history=history_summary(logs),
            last_readout=logs[-1]["readout_text"] if logs else "No prior yet.",
            hotspots=hotspot_summary(X_obs, y_obs),
        )

        readout = agent.propose(prompt)
        prior_surface = evaluate_prior_surface(readout, X1, X2)
        metrics = alignment_metrics(truth, prior_surface)

        logs.append(
            {
                "iteration": iteration,
                "n_samples": len(y_obs),
                "rho": metrics["rho"],
                "rmse": metrics["rmse"],
                "alpha": metrics["alpha"],
                "label": readout.get("label", readout.get("description", "LLM prior")),
                "readout": readout,
                "readout_text": summarize_readout(readout),
            }
        )

        print(
            f"[Iter {iteration}] Samples={len(y_obs):3d} | rho={metrics['rho']:.3f} "
            f"| rmse={metrics['rmse']:.3f} | alpha={metrics['alpha']:.3f}"
        )
        if fig_dir or show_figs:
            fig_path = (fig_dir / f"iteration_{iteration:02d}.png") if fig_dir else None
            render_iteration_figure(
                iteration=iteration,
                X1=X1,
                X2=X2,
                truth=truth,
                prior=prior_surface,
                X_obs=X_obs,
                y_obs=y_obs,
                save_path=fig_path,
                show=show_figs,
            )

    return logs


def save_logs(logs: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)
    print(f"[llm_prior_benchmark] Saved log to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative LLM prior benchmark on the Franke surface.")
    parser.add_argument("--iterations", type=int, default=8, help="Number of LLM prior proposals.")
    parser.add_argument("--batch-size", type=int, default=5, help="Samples added per iteration.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument("--use-llm", action="store_true", help="Enable real LLM calls (requires OPENAI_API_KEY).")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI model name.")
    parser.add_argument("--temperature", type=float, default=0.2, help="LLM temperature.")
    parser.add_argument("--log-json", type=str, default=None, help="Optional path to save alignment logs.")
    parser.add_argument("--fig-dir", type=str, default=None, help="Directory to save iteration figures.")
    parser.add_argument("--show-figs", action="store_true", default=False, help="Display iteration figures interactively.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs = run_benchmark(
        iterations=args.iterations,
        batch_size=args.batch_size,
        seed=args.seed,
        use_llm=args.use_llm,
        model=args.model,
        temperature=args.temperature,
        fig_dir=Path(args.fig_dir) if args.fig_dir else None,
        show_figs=args.show_figs,
    )
    if args.log_json:
        save_logs(logs, args.log_json)


# logs = run_benchmark(
#     iterations=8,
#     batch_size=5,
#     seed=0,
#     use_llm=True,   # set True if OPENAI_API_KEY is configured
#     model="gpt-4o-mini",
#     temperature=0.2,
# )

# if __name__ == "__main__":
#     main()


# Example interactive usage (run this cell inside an IPython/VS Code notebook).
logs = run_benchmark(
    iterations=20,
    batch_size=5,
    seed=6,
    use_llm=True,   # set True if OPENAI_API_KEY is configured
    model="gpt-4o-mini",
    temperature=0.2,
    fig_dir=None,
    show_figs=True,
)
#%%
