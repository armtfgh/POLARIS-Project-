

"""
Lookup‑Table Optimization on Arbitrary CSV (No synthetic function)
==================================================================

What this script provides
-------------------------
- **Input**: a CSV where **all columns except the last** are numeric features (d dims), and the **last column** is the objective (yield).
- **Domain**: the CSV rows themselves are the **finite candidate set**. No fitted surrogate oracle for the true function; the table is the ground truth.
- **Methods** (sequential over the table):
  1) **Random**: pick unseen rows uniformly at random.
  2) **Baseline BO (EI)**: fit a GP **only on the already‑revealed rows**, score **EI on the remaining rows**, choose the best.
  3) **Hybrid (prior + residual GP)**: like Baseline BO but allows a (flat by default) **prior mean** shaped by a JSON readout (LLM optional). Uses a residual GP over the prior.

New: Repetitions & Plotting
---------------------------
- Each main method now accepts a **`repeats`** parameter. Runs are repeated with seeds `seed, seed+1, ..., seed+repeats-1`.
- Use `compare_methods_from_csv(..., repeats=k)` to run all methods across `k` seeds.
- `plot_runs_mean_lookup(...)` plots **mean best‑so‑far** with a shaded uncertainty band across seeds.

Usage (Jupyter)
----------------
```python
# 1) Load the CSV
lt = load_lookup_csv("my_experiment_data.csv", impute_features="median")

# 2) Compare methods across 5 seeds and plot
hist = compare_methods_from_csv(lt, n_init=6, n_iter=25, seed=0, repeats=5,
                                include_hybrid=True, readout_source="flat")
plot_runs_mean_lookup(hist, ci="sem")  # or ci="sd" or "95ci"
```

Dependencies
------------
- torch >= 1.11, botorch, gpytorch, numpy, pandas, matplotlib
"""

from __future__ import annotations

# %%
import warnings
warnings.filterwarnings('ignore')

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Any, List, Optional, Tuple

import os, json
import numpy as np
import pandas as pd

import torch
from torch import Tensor

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.optim.optimize import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples

import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood

import matplotlib.pyplot as plt

# Optional LLM client (only used if readout_source="llm")

try:
    import httpx
    from openai import OpenAI
    _OPENAI_CLIENT = OpenAI(http_client=httpx.Client(verify=False))
except Exception:
    _OPENAI_CLIENT = None


# -------------------- Device / dtype --------------------
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
DTYPE = torch.float32
torch.set_default_dtype(DTYPE)

# -------------------- Prior / residual machinery --------------------
from data_analysis import build_ugi_ml_oracle, RandomForestOracle
from prior_gp import alignment_on_obs, fit_residual_gp, GPWithPriorMean, Prior
from readout_schema import readout_to_prior, flat_readout

# (Prompts are optional; only used if readout_source="llm")
#%%
from bo_readout_prompts import (
    SYS_PROMPTS_PERFECT,
    SYS_PROMPTS_GOOD,
    SYS_PROMPTS_MEDIUM,
    SYS_PROMPTS_RANDOM,
    SYS_PROMPTS_BAD,
    SYS_PROMPT_MINIMAL_HUMAN,
    SYS_PROMPTS_CUSTOM,
    SYS_PROMPTS_BEST,
)

PROMPT_LIBRARY = {
    "perfect": SYS_PROMPTS_PERFECT,
    "good": SYS_PROMPTS_GOOD,
    "medium": SYS_PROMPTS_MEDIUM,
    "minimal": SYS_PROMPT_MINIMAL_HUMAN,
    "random": SYS_PROMPTS_RANDOM,
    "bad": SYS_PROMPTS_BAD,
    "custom": SYS_PROMPTS_CUSTOM,
    "best": SYS_PROMPTS_BEST,
}

# Global control so every campaign uses the same Sobol initialization batch.
# Leave as None for per-run randomness; set to an int only if you explicitly
# want all campaigns (and seeds) to start from the exact same Sobol picks.
SHARED_SOBOL_INIT_SEED: Optional[int] = None

#%%
# ====================================================================
#                             DATASET LOADER
# ====================================================================


#%%
@dataclass
class LookupTable:
    X_raw: Tensor          # (N, d) raw (unnormalized) features
    y: Tensor              # (N,) objective (float)
    X: Tensor              # (N, d) normalized to [0,1]
    mins: Tensor           # (d,)
    maxs: Tensor           # (d,)
    feature_names: List[str]
    objective_name: str
    oracle: Optional[Callable[[Tensor], Tensor]] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def d(self) -> int:
        return int(self.X.shape[1])


@dataclass
class ContinuousDomain:
    feature_names: List[str]
    mins: Tensor
    maxs: Tensor
    bounds: Tensor  # raw bounds
    unit_bounds: Tensor  # [0,1]^d bounds for optimisation
    oracle: RandomForestOracle
    metadata: Dict[str, Any]


def load_lookup_csv(path: str,
                     *,
                     device: torch.device = DEVICE,
                     dtype: torch.dtype = DTYPE,
                     impute_features: Optional[str] = None,  # None | 'median'
                     aggregate_duplicates: Optional[str] = "mean"  # None | 'mean' | 'median' | 'max'
                     ) -> LookupTable:
    """Load a CSV as a finite candidate set.

    Assumptions
    -----------
    - All columns except the last are numeric features; the last column is the objective.
    - Rows may contain NaN/inf or non-numeric strings; we coerce to numeric and clean deterministically.

    Cleaning policy
    ---------------
    - We always DROP rows with invalid objective (NaN/inf).
    - For features:
        * impute_features == 'median' -> fill NaN/inf with column median (computed per feature).
        * impute_features is None     -> DROP rows where any feature is NaN/inf.
    - If `aggregate_duplicates` is not None (default "mean"), rows with identical features are
      collapsed using the chosen statistic on the objective. This enforces the deterministic-table
      assumption for lookup BO.
    """
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least 2 columns (>=1 feature + 1 objective)")

    feat_cols = list(df.columns[:-1])
    obj_col = df.columns[-1]

    # Coerce to numeric and replace inf with NaN
    for c in feat_cols + [obj_col]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    df[obj_col] = df[obj_col].replace([np.inf, -np.inf], np.nan)

    # Always drop rows with invalid objective
    mask_y = df[obj_col].notna()

    if impute_features == 'median':
        med = df.loc[mask_y, feat_cols].median(numeric_only=True)
        df.loc[:, feat_cols] = df[feat_cols].fillna(med)
        mask_x = df[feat_cols].notna().all(axis=1)
    else:
        mask_x = df[feat_cols].notna().all(axis=1)

    mask = (mask_y & mask_x)
    df_clean = df.loc[mask].reset_index(drop=True)

    if df_clean.shape[0] < 2:
        raise ValueError("After cleaning, fewer than 2 valid rows remain. Please inspect your CSV or enable median imputation for features.")

    dup_mask = df_clean.duplicated(subset=feat_cols, keep=False)
    if aggregate_duplicates:
        agg = aggregate_duplicates.lower()
        if agg not in {"mean", "median", "max"}:
            raise ValueError("aggregate_duplicates must be one of {'mean','median','max', None}")
        if dup_mask.any():
            dup_count = int(dup_mask.sum())
            if agg == "mean":
                df_clean = df_clean.groupby(feat_cols, as_index=False)[obj_col].mean()
            elif agg == "median":
                df_clean = df_clean.groupby(feat_cols, as_index=False)[obj_col].median()
            else:  # agg == "max"
                df_clean = df_clean.groupby(feat_cols, as_index=False)[obj_col].max()
            print(f"[load_lookup_csv] Aggregated {dup_count} duplicate rows using '{agg}'.")

    X_raw_np = df_clean[feat_cols].to_numpy(dtype=np.float64)
    y_np = df_clean[obj_col].to_numpy(dtype=np.float64)

    X_raw = torch.tensor(X_raw_np, dtype=dtype, device=device)
    y = torch.tensor(y_np, dtype=dtype, device=device).reshape(-1)

    mins = X_raw.min(dim=0).values
    maxs = X_raw.max(dim=0).values
    rng = (maxs - mins).clamp_min(1e-12)
    X = (X_raw - mins) / rng

    return LookupTable(
        X_raw=X_raw, y=y, X=X, mins=mins, maxs=maxs,
        feature_names=feat_cols, objective_name=str(obj_col)
    )


def _make_unit_to_raw_fn(mins: Tensor, maxs: Tensor) -> Callable[[Tensor], Tensor]:
    rng = (maxs - mins).clamp_min(1e-12)

    def _convert(x: Tensor) -> Tensor:
        while x.ndim < mins.ndim:
            x = x.unsqueeze(0)
        return mins.to(device=x.device, dtype=x.dtype) + x * rng.to(device=x.device, dtype=x.dtype)

    return _convert


def build_continuous_domain(
    *,
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
) -> ContinuousDomain:
    payload = build_ugi_ml_oracle()
    candidate_pool: pd.DataFrame = payload["candidate_pool_df"]  # type: ignore[index]
    feature_names: List[str] = payload["feature_columns"]  # type: ignore[index]

    mins_np = candidate_pool.min().to_numpy(dtype="float64")
    maxs_np = candidate_pool.max().to_numpy(dtype="float64")
    mins = torch.tensor(mins_np, dtype=torch.double)
    maxs = torch.tensor(maxs_np, dtype=torch.double)
    bounds_raw = torch.stack([mins.to(dtype=dtype), maxs.to(dtype=dtype)])
    unit_bounds = torch.stack(
        [
            torch.zeros(len(feature_names), dtype=dtype, device=DEVICE),
            torch.ones(len(feature_names), dtype=dtype, device=DEVICE),
        ],
        dim=0,
    )

    rf_model = payload["model"]  # type: ignore[index]
    rf_oracle = RandomForestOracle(rf_model, feature_names)

    metadata = {
        "oracle_metrics": payload.get("metrics"),
        "feature_importances": payload.get("feature_importances"),
        "n_training_rows": len(candidate_pool),
    }

    return ContinuousDomain(
        feature_names=feature_names,
        mins=mins,
        maxs=maxs,
        bounds=bounds_raw.to(dtype=dtype),
        unit_bounds=unit_bounds,
        oracle=rf_oracle,
        metadata=metadata,
    )


def unit_to_raw(domain: ContinuousDomain, X_unit: Tensor) -> Tensor:
    mins = domain.mins.to(device=X_unit.device, dtype=X_unit.dtype)
    rng = (domain.maxs - domain.mins).to(device=X_unit.device, dtype=X_unit.dtype).clamp_min(1e-12)
    return mins + X_unit * rng


def evaluate_oracle(domain: ContinuousDomain, X_unit: Tensor) -> Tensor:
    raw = unit_to_raw(domain, X_unit)
    return domain.oracle(raw)


def _record_continuous_sample(
    domain: ContinuousDomain,
    x_unit: Tensor,
    y: float,
    best: float,
    method: str,
    iteration: int,
) -> Dict[str, Any]:
    raw = unit_to_raw(domain, x_unit.detach())
    rec: Dict[str, Any] = {
        "iter": iteration,
        "method": method,
        "y": float(y),
        "best_so_far": float(best),
    }
    for idx, value in enumerate(raw.tolist()):
        rec[f"x{idx+1}"] = float(value)
    return rec


def run_random_continuous(
    domain: ContinuousDomain,
    *,
    n_init: int,
    n_iter: int,
    seed: int = 0,
) -> pd.DataFrame:
    total = n_init + n_iter
    samples = draw_sobol_samples(bounds=domain.unit_bounds, n=total, q=1, seed=seed).squeeze(1)
    recs: List[Dict[str, Any]] = []
    best = float("-inf")

    for i in range(n_init):
        x = samples[i]
        y = float(evaluate_oracle(domain, x).item())
        best = max(best, y)
        recs.append(
            _record_continuous_sample(domain, x, y, best, method="random", iteration=i - n_init)
        )

    for t in range(n_iter):
        x = draw_sobol_samples(bounds=domain.unit_bounds, n=1, q=1, seed=seed + 4242 + t).squeeze(1)[0]
        y = float(evaluate_oracle(domain, x).item())
        best = max(best, y)
        recs.append(_record_continuous_sample(domain, x, y, best, method="random", iteration=t))

    return pd.DataFrame(recs)


def run_baseline_ei_continuous(
    domain: ContinuousDomain,
    *,
    n_init: int,
    n_iter: int,
    seed: int = 0,
    num_restarts: int = 15,
    raw_samples: int = 512,
) -> pd.DataFrame:
    X_init = draw_sobol_samples(bounds=domain.unit_bounds, n=n_init, q=1, seed=seed).squeeze(1)
    Y_init = evaluate_oracle(domain, X_init).unsqueeze(-1)

    X_obs = X_init.clone()
    Y_obs = Y_init.clone()
    recs: List[Dict[str, Any]] = []
    best = float(Y_obs.max().item()) if Y_obs.numel() else float("-inf")

    for i in range(n_init):
        y = float(Y_init[i].item())
        best = max(best, y)
        recs.append(
            _record_continuous_sample(domain, X_init[i], y, best, method="baseline_ei", iteration=i - n_init)
        )

    for t in range(n_iter):
        gp = SingleTaskGP(X_obs, Y_obs)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        best_f = float(Y_obs.max().item()) if Y_obs.numel() else 0.0
        EI = ExpectedImprovement(model=gp, best_f=best_f, maximize=True)
        x_next, _ = optimize_acqf(
            EI,
            bounds=domain.unit_bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        x_next = x_next.squeeze(0)
        y_next = evaluate_oracle(domain, x_next).unsqueeze(-1)

        X_obs = torch.cat([X_obs, x_next.unsqueeze(0)], dim=0)
        Y_obs = torch.cat([Y_obs, y_next], dim=0)

        y_value = float(y_next.item())
        best = max(best, y_value)
        recs.append(_record_continuous_sample(domain, x_next, y_value, best, method="baseline_ei", iteration=t))

    return pd.DataFrame(recs)


def _sample_unit_pool(bounds: Tensor, n: int, seed: int) -> Tensor:
    return draw_sobol_samples(bounds=bounds, n=n, q=1, seed=seed).squeeze(1)


def run_hybrid_continuous(
    domain: ContinuousDomain,
    *,
    n_init: int,
    n_iter: int,
    seed: int = 0,
    readout_source: str = "flat",
    prompt_profile: str = "perfect",
    model: str = "gpt-4o-mini",
    pool_size: int = 2048,
    num_restarts: int = 15,
    raw_samples: int = 512,
    prior_strength: float = 1.0,
    rho_floor: float = 0.05,
    diagnose_prior: bool = False,
    early_prior_boost: bool = False,
    early_prior_steps: int = 5,
) -> pd.DataFrame:
    X_init = draw_sobol_samples(bounds=domain.unit_bounds, n=n_init, q=1, seed=seed).squeeze(1)
    Y_init = evaluate_oracle(domain, X_init).unsqueeze(-1)

    X_obs = X_init.clone()
    Y_obs = Y_init.clone()
    recs: List[Dict[str, Any]] = []
    best = float(Y_obs.max().item()) if Y_obs.numel() else float("-inf")

    unit_to_raw_fn = _make_unit_to_raw_fn(domain.mins, domain.maxs)

    for i in range(n_init):
        y = float(Y_init[i].item())
        best = max(best, y)
        recs.append(
            _record_continuous_sample(domain, X_init[i], y, best, method="hybrid", iteration=i - n_init)
        )

    if readout_source == "llm":
        prompt_text = PROMPT_LIBRARY.get(prompt_profile)
        pool = _sample_unit_pool(domain.unit_bounds, max(pool_size, 512), seed + 101)
        Y_obs2 = Y_obs
        gp_ctx = SingleTaskGP(X_obs, Y_obs2)
        mll_ctx = ExactMarginalLogLikelihood(gp_ctx.likelihood, gp_ctx)
        fit_gpytorch_mll(mll_ctx)
        ro0_raw = llm_generate_readout(
            pd.DataFrame(recs),
            gp_ctx,
            X_obs,
            Y_obs2,
            pool,
            sys_prompt=prompt_text,
            model=model,
            unit_to_raw_fn=unit_to_raw_fn,
        )
        print(f"[Hybrid][continuous] seed={seed} readout ({prompt_profile}):\n"
              f"{json.dumps(ro0_raw, indent=2)}")
        ro0_sane = sanitize_readout_dim(ro0_raw, len(domain.feature_names), feature_names=domain.feature_names)
        ro0 = _normalize_readout_to_unit_box(ro0_sane, domain.mins, domain.maxs, feature_names=domain.feature_names)
    else:
        ro0 = flat_readout(feature_names=domain.feature_names)

    prior = readout_to_prior(ro0, feature_names=domain.feature_names)
    prior_debug: Optional[List[Dict[str, Any]]] = [] if diagnose_prior else None

    for t in range(n_iter):
        pool = _sample_unit_pool(domain.unit_bounds, max(pool_size, 1024), seed + 505 + t)

        if early_prior_boost and t < early_prior_steps:
            prior_vals = prior.m0_torch(pool).reshape(-1)
            idx_local = int(torch.argmax(prior_vals))
            x_next = pool[idx_local]
            y_next = evaluate_oracle(domain, x_next).unsqueeze(-1)
            X_obs = torch.cat([X_obs, x_next.unsqueeze(0)], dim=0)
            Y_obs = torch.cat([Y_obs, y_next], dim=0)
            y_value = float(y_next.item())
            best = max(best, y_value)
            rec = _record_continuous_sample(domain, x_next, y_value, best, method="hybrid", iteration=t)
            rec["prior_boost"] = True
            recs.append(rec)
            continue

        gp_resid, alpha = fit_residual_gp(X_obs, Y_obs, prior)
        rho = alignment_on_obs(X_obs, Y_obs, prior)
        rho_weight = max(abs(float(rho)), rho_floor)
        m0_scale = float(alpha * prior_strength * rho_weight)
        model_total = GPWithPriorMean(gp_resid, prior, m0_scale=m0_scale)

        EI = ExpectedImprovement(model=model_total, best_f=float(Y_obs.max().item()), maximize=True)
        x_next, _ = optimize_acqf(
            EI,
            bounds=domain.unit_bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        x_next = x_next.squeeze(0)
        y_next = evaluate_oracle(domain, x_next).unsqueeze(-1)

        X_obs = torch.cat([X_obs, x_next.unsqueeze(0)], dim=0)
        Y_obs = torch.cat([Y_obs, y_next], dim=0)

        y_value = float(y_next.item())
        best = max(best, y_value)
        recs.append(_record_continuous_sample(domain, x_next, y_value, best, method="hybrid", iteration=t))

        if diagnose_prior and prior_debug is not None:
            gp_plain = SingleTaskGP(X_obs, Y_obs)
            mll_plain = ExactMarginalLogLikelihood(gp_plain.likelihood, gp_plain)
            fit_gpytorch_mll(mll_plain)
            EI_plain = ExpectedImprovement(model=gp_plain, best_f=float(Y_obs.max().item()), maximize=True)
            x_plain, _ = optimize_acqf(
                EI_plain,
                bounds=domain.unit_bounds,
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )
            x_plain = x_plain.squeeze(0)
            pool_diag = _sample_unit_pool(domain.unit_bounds, max(pool_size, 1024), seed + 202 + t)
            prior_vals = prior.m0_torch(pool_diag).reshape(-1)
            idx_prior = int(torch.argmax(prior_vals))
            x_prior = pool_diag[idx_prior]

            prior_debug.append(
                {
                    "iter": t,
                    "hybrid_point": [float(v) for v in unit_to_raw(domain, x_next).tolist()],
                    "baseline_point": [float(v) for v in unit_to_raw(domain, x_plain).tolist()],
                    "prior_point": [float(v) for v in unit_to_raw(domain, x_prior).tolist()],
                    "alpha": float(alpha),
                    "rho": float(rho),
                    "rho_weight": float(rho_weight),
                    "m0_scale": float(m0_scale),
                }
            )

    df = pd.DataFrame(recs)
    if diagnose_prior and prior_debug is not None:
        df.attrs["prior_debug"] = prior_debug
    return df


def compare_methods_continuous(
    domain: ContinuousDomain,
    *,
    n_init: int = 6,
    n_iter: int = 25,
    seed: int = 0,
    repeats: int = 1,
    include_hybrid: bool = True,
    readout_source: str = "flat",
    prompt_profiles: Optional[List[str]] = None,
    prior_strength: float = 1.0,
    rho_floor: float = 0.05,
    diagnose_prior: bool = False,
    early_prior_boost: bool = False,
    early_prior_steps: int = 5,
) -> pd.DataFrame:
    if isinstance(prompt_profiles, str):
        prompt_profiles = [prompt_profiles]
    prompt_profiles = prompt_profiles or ["perfect"]
    dfs: List[pd.DataFrame] = []

    prior_debug_runs: List[Dict[str, Any]] = [] if diagnose_prior else []

    for r in range(repeats):
        current_seed = seed + r

        rand = run_random_continuous(domain, n_init=n_init, n_iter=n_iter, seed=current_seed)
        rand["seed"] = current_seed
        dfs.append(rand)

        base = run_baseline_ei_continuous(domain, n_init=n_init, n_iter=n_iter, seed=current_seed)
        base["seed"] = current_seed
        dfs.append(base)

        if include_hybrid:
            if readout_source == "llm":
                for profile in prompt_profiles:
                    method_label = f"hybrid_{profile}"
                    print(f"[Hybrid] continuous seed={current_seed} using readout '{profile}'")
                    hyb = run_hybrid_continuous(
                        domain,
                        n_init=n_init,
                        n_iter=n_iter,
                        seed=current_seed,
                        readout_source="llm",
                        prompt_profile=profile,
                        prior_strength=prior_strength,
                        rho_floor=rho_floor,
                        diagnose_prior=diagnose_prior,
                        early_prior_boost=early_prior_boost,
                        early_prior_steps=early_prior_steps,
                    )
                    hyb["seed"] = current_seed
                    hyb["method"] = method_label
                    if diagnose_prior and "prior_debug" in hyb.attrs:
                        prior_debug_runs.append(
                            {"seed": current_seed, "prior_debug": hyb.attrs["prior_debug"]}
                        )
                    dfs.append(hyb)
            else:
                print(f"[Hybrid] continuous seed={current_seed} using flat readout")
                hyb = run_hybrid_continuous(
                    domain,
                    n_init=n_init,
                    n_iter=n_iter,
                    seed=current_seed,
                    readout_source=readout_source,
                    prompt_profile="perfect",
                    prior_strength=prior_strength,
                    rho_floor=rho_floor,
                    diagnose_prior=diagnose_prior,
                    early_prior_boost=early_prior_boost,
                    early_prior_steps=early_prior_steps,
                )
                hyb["seed"] = current_seed
                hyb["method"] = "hybrid_flat"
                if diagnose_prior and "prior_debug" in hyb.attrs:
                    prior_debug_runs.append(
                        {"seed": current_seed, "prior_debug": hyb.attrs["prior_debug"]}
                    )
                dfs.append(hyb)

    out = pd.concat(dfs, ignore_index=True)
    if diagnose_prior and prior_debug_runs:
        out.attrs["prior_debug_runs"] = prior_debug_runs
    return out


def build_lookup_from_ugi_oracle(
    *,
    save_merged_path: Optional[str] = None,
    oracle_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[LookupTable, Dict[str, Any]]:
    """Train the RandomForest oracle on UGI data and expose it as a LookupTable."""

    kwargs = dict(oracle_kwargs or {})
    if save_merged_path:
        kwargs["save_merged_path"] = Path(save_merged_path)

    payload = build_ugi_ml_oracle(**kwargs)

    feature_columns: List[str] = payload["feature_columns"]  # type: ignore[index]
    candidate_pool_df: pd.DataFrame = payload["candidate_pool_df"]  # type: ignore[index]
    oracle = payload["oracle"]  # type: ignore[index]
    metrics = payload["metrics"]  # type: ignore[index]
    importances = payload["feature_importances"]  # type: ignore[index]

    candidate_np = candidate_pool_df.to_numpy(dtype=np.float64)
    X_raw = torch.tensor(candidate_np, dtype=DTYPE, device=DEVICE)
    mins = X_raw.min(dim=0).values
    maxs = X_raw.max(dim=0).values
    rng = (maxs - mins).clamp_min(1e-12)
    X = (X_raw - mins) / rng

    # Evaluate oracle on CPU (scikit-learn lives on CPU)
    X_cpu = torch.tensor(candidate_np, dtype=torch.double)
    with torch.no_grad():
        preds = oracle(X_cpu).reshape(-1)
    y = preds.to(dtype=DTYPE, device=DEVICE)

    lookup = LookupTable(
        X_raw=X_raw,
        y=y,
        X=X,
        mins=mins,
        maxs=maxs,
        feature_names=feature_columns,
        objective_name="predicted_yield",
        oracle=oracle,
        metadata={
            "oracle_metrics": metrics,
            "feature_importances": importances,
            "n_candidates": int(X_raw.shape[0]),
        },
    )

    return lookup, {
        "dataframe": payload["dataframe"],  # type: ignore[index]
        "candidate_pool_df": candidate_pool_df,
    }

# ====================================================================
#                         HELPER UTILITIES
# ====================================================================

def select_initial_indices(n_total: int, n_init: int, seed: int) -> List[int]:
    g = torch.Generator(device='cpu')
    g.manual_seed(int(seed))
    perm = torch.randperm(n_total, generator=g).tolist()
    return perm[:max(1, min(n_init, n_total))]


def remaining_indices(n_total: int, seen: set[int]) -> List[int]:
    return [i for i in range(n_total) if i not in seen]


def as_records_row(idx: int, x_feat: Tensor, y: float, best: float, method: str, feature_names: List[str]) -> Dict[str, Any]:
    rec = {
        "iter": None,  # filled by caller
        "idx": int(idx),
        "y": float(y),
        "best_so_far": float(best),
        "method": method,
    }
    for j in range(x_feat.numel()):
        rec[f"x{j+1}"] = float(x_feat[j].item())
    return rec


try:
    from llm_si import ExperimentState, run_llm_scout_init
except ImportError:  # pragma: no cover - optional dependency
    ExperimentState = None  # type: ignore[assignment]
    run_llm_scout_init = None  # type: ignore[assignment]

# ---------- Initialization helpers ----------

def _sobol_init_lookup(lookup, n_init: int, seed: int) -> List[int]:
    """
    Generate n_init Sobol points in [0,1]^d, map them to the RAW feature scale
    using lookup.mins/maxs, then snap each proposal to the nearest unseen
    candidate (Euclidean distance in RAW space). Ensures uniqueness across
    picks and diversifies across seeds via a fast_forward offset.
    """
    from torch.quasirandom import SobolEngine

    d, N = lookup.d, lookup.n
    if n_init <= 0:
        return []

    engine = SobolEngine(dimension=d, scramble=True, seed=int(seed))
    # Seed-dependent offset so repeats differ even with same n_init
    offset = (int(seed) % 10007) * 64
    if offset:
        engine.fast_forward(offset)

    props_unit = engine.draw(n_init).to(device=lookup.X_raw.device, dtype=lookup.X_raw.dtype)  # (n_init,d)

    # Map unit-box proposals to RAW feature scale: x_raw = mins + u * (maxs - mins)
    mins = lookup.mins.to(device=lookup.X_raw.device, dtype=lookup.X_raw.dtype)
    rng = (lookup.maxs - lookup.mins).to(device=lookup.X_raw.device, dtype=lookup.X_raw.dtype).clamp_min(1e-12)
    props_raw = mins + props_unit * rng

    remaining = set(range(N))
    chosen: List[int] = []
    for k in range(n_init):
        x_prop = props_raw[k:k+1]  # (1,d) RAW
        # compute distances ONLY over remaining
        rem_idx = torch.tensor(sorted(list(remaining)), device=lookup.X_raw.device, dtype=torch.long)
        cand = lookup.X_raw.index_select(0, rem_idx)
        j = int(torch.argmin(torch.linalg.norm(cand - x_prop, dim=1)).item())
        idx = int(rem_idx[j].item())
        chosen.append(idx)
        remaining.remove(idx)

    return chosen

def _llm_si_init_lookup(lookup, n_init: int) -> List[int]:
    """
    Run LLM-SI on the discrete table (normalized space). Returns used indices.
    """
    if ExperimentState is None or run_llm_scout_init is None:
        raise RuntimeError(
            "llm_si module is not available in this environment; "
            "install it or disable 'llm' initialisation"
        )

    all_idx = list(range(lookup.n))
    state = ExperimentState(
        X_obs=torch.empty((0, lookup.d), device=lookup.X.device, dtype=lookup.X.dtype),
        y_obs=torch.empty((0,), device=lookup.X.device, dtype=lookup.X.dtype),
        log=[],
        bounds=None,                          # we are in lookup mode
        candidate_pool=lookup.X,              # normalized features
        candidate_y=lookup.y,                 # ground-truth objective
        remaining_idx=all_idx.copy(),         # start with all available
    )
    # Run the scouting phase for n_init picks
    state = run_llm_scout_init(state, n_scout=int(n_init))
    # The used indices are simply what's not remaining
    used = sorted(set(all_idx) - set(state.remaining_idx))
    return used


def _select_initial_indices_lookup(lookup: LookupTable, n_init: int, seed: int, init_method: str) -> List[int]:
    """
    Shared initializer selector so different algorithms can start from identical seeds.
    """
    method = (init_method or "random").lower()
    if method in ("random", "uniform"):
        return select_initial_indices(lookup.n, n_init, seed)
    if method in ("sobol", "sobo"):
        sobol_seed = SHARED_SOBOL_INIT_SEED if SHARED_SOBOL_INIT_SEED is not None else seed
        return _sobol_init_lookup(lookup, n_init, sobol_seed)
    if method in ("llm-si", "llm_si", "llmsi"):
        return _llm_si_init_lookup(lookup, n_init)
    raise ValueError(f"Unknown init_method: {init_method}")


def _build_initial_from_indices(lookup, idxs: List[int], method_tag: str) -> tuple[set[int], torch.Tensor, torch.Tensor, list]:
    """
    Given a list of selected indices, build seen set, X_obs, Y_obs and
    the initial records with negative iteration ids (matching your existing convention).
    """
    seen = set(int(i) for i in idxs)
    X_obs = lookup.X[idxs]
    Y_obs = lookup.y[idxs]
    rec = []
    best_running = float('-inf')
    for i, idx in enumerate(idxs):
        y_i = float(lookup.y[idx].item())
        best_running = max(best_running, y_i)
        row = as_records_row(idx, lookup.X_raw[idx], y_i, best_running, method=method_tag, feature_names=lookup.feature_names)
        row["iter"] = i - len(idxs)
        rec.append(row)
    return seen, X_obs, Y_obs, rec

# ====================================================================
#                     SUMMARIES FOR (OPTIONAL) LLM READOUT
# ====================================================================

def summarize_on_pool_for_llm(
    gp: SingleTaskGP,
    X_obs: Tensor,
    Y_obs: Tensor,
    X_pool: Tensor,
    topk: int = 12,
    unit_to_raw_fn: Optional[Callable[[Tensor], Tensor]] = None,
) -> Dict[str, Any]:
    """Summarize model state on a discrete pool for prompting.
    Handles Y_obs shaped (n,) or (n,1). Returns top EI / top variance points and incumbent.
    """
    if Y_obs.numel() == 0 or X_obs.numel() == 0:
        def _convert(vec: Tensor) -> List[float]:
            if unit_to_raw_fn is not None:
                vec = unit_to_raw_fn(vec)
            return [float(v) for v in vec.detach().cpu().tolist()]

        k = int(min(topk, X_pool.shape[0]))
        entries = [
            {"x": _convert(X_pool[i]), "score": None}
            for i in range(k)
        ]
        incumbent = {"x": _convert(X_pool[0]) if X_pool.shape[0] else [], "y": None}
        return {
            "best_so_far": None,
            "incumbent": incumbent,
            "top_ei": entries,
            "top_var": [{"x": e["x"], "var": None} for e in entries],
        }

    with torch.no_grad():
        Y_flat = Y_obs.squeeze(-1) if Y_obs.dim() == 2 else Y_obs
        best_f = float(Y_flat.max().item())
        EI = ExpectedImprovement(model=gp, best_f=best_f, maximize=True)
        ei_vals = EI(X_pool.unsqueeze(1)).reshape(-1)
        post = gp.posterior(X_pool.unsqueeze(1))
        var = post.variance.reshape(-1)

        k = int(min(topk, X_pool.shape[0])) if X_pool.shape[0] > 0 else 0
        idx_ei = torch.topk(ei_vals, k=k).indices if k > 0 else torch.tensor([], dtype=torch.long)
        idx_var = torch.topk(var, k=k).indices if k > 0 else torch.tensor([], dtype=torch.long)

        def _convert(vec: Tensor) -> List[float]:
            if unit_to_raw_fn is not None:
                vec = unit_to_raw_fn(vec)
            return [float(v) for v in vec.detach().cpu().tolist()]

        def pack(idx):
            out = []
            for i in idx:
                xi = X_pool[int(i)].detach().cpu().tolist()
                vec = torch.tensor(xi, dtype=X_pool.dtype, device=X_pool.device)
                out.append({"x": _convert(vec), "score": float(ei_vals[int(i)].item())})
            return out

        inc_i = int(torch.argmax(Y_flat))
        inc = {"x": _convert(X_obs[inc_i]), "y": float(Y_flat[inc_i].item())}

        return {"best_so_far": best_f, "incumbent": inc,
                "top_ei": pack(idx_ei),
                "top_var": [
                    {"x": _convert(X_pool[int(i)]), "var": float(var[int(i)].item())}
                    for i in idx_var
                ],}

# ---- Readout dimension sanitizer ----------------------------------

def _coerce_list_len(vals, d, fill=0.5):
    if isinstance(vals, (int, float)):
        vals = [float(vals)]
    vals = list(vals)
    if len(vals) >= d:
        return [float(v) for v in vals[:d]]
    else:
        return [float(v) for v in (vals + [fill] * (d - len(vals)))]


def sanitize_readout_dim(readout: Dict[str, Any], d: int, *, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """Coerce readout JSON to match feature dimension d.
    - Bumps: pad/truncate mu to length d; sigma scalar ok or list padded/truncated to d; keep amp as float.
    - Effects: keep only keys x1..xd.
    - Interactions: drop entries that refer to out-of-range dims when integer indices are provided.
    """
    if readout is None:
        return {"effects": {}, "interactions": [], "bumps": []}
    ro = {**readout}

    # bumps
    bumps_in = ro.get("bumps", []) or []
    bumps_out = []
    for b in bumps_in:
        
        mu = _coerce_list_len(b.get("mu", [0.5] * d), d, fill=0.5)
        sigma = b.get("sigma", 0.15)
        if isinstance(sigma, (list, tuple)):
            sigma = _coerce_list_len(sigma, d, fill=(sigma[-1] if len(sigma) else 0.15))
        amp = float(b.get("amp", 0.1))
        bumps_out.append({"mu": mu, "sigma": sigma, "amp": amp})

    ro["bumps"] = bumps_out

    # effects: keep only x1..xd
    allowed_named = {name: idx for idx, name in enumerate(feature_names or [])}
    allowed_lower = {name.lower(): idx for idx, name in enumerate(feature_names or [])}

    def _dim_for_key(key: str) -> Optional[int]:
        if key.startswith("x"):
            try:
                j = int(key[1:]) - 1
            except ValueError:
                return None
            return j if 0 <= j < d else None
        if key in allowed_named:
            return allowed_named[key]
        low = key.lower()
        return allowed_lower.get(low)

    eff_in = ro.get("effects", {}) or {}
    eff_out = {}
    for k, v in eff_in.items():
        if isinstance(k, str):
            idx = _dim_for_key(k)
            if idx is not None:
                eff_out[k] = v

    ro["effects"] = eff_out

    # interactions: conservative filter
    inter_in = ro.get("interactions", []) or []
    inter_out = []
    for it in inter_in:
        keep = True
        pair = it.get("pair") or it.get("vars") or it.get("indices")
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            a, b = pair
            targets = []
            for val in (a, b):
                if isinstance(val, int):
                    targets.append(1 <= val <= d)
                elif isinstance(val, str):
                    targets.append(_dim_for_key(val) is not None)
                else:
                    targets.append(False)
            if not all(targets):
                keep = False
        if keep:
            inter_out.append(it)
    ro["interactions"] = inter_out
    return ro


def _normalize_readout_to_unit_box(readout: Dict[str, Any], mins: Tensor, maxs: Tensor,
                                   *, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Take raw-scale readout JSON (mu/sigma in original units) and convert to unit-box coordinates.
    Sigma entries may be either scalar or list; scalars are scaled by the average range.
    """
    if readout is None:
        return {"effects": {}, "interactions": [], "bumps": []}
    rng = (maxs - mins).clamp_min(1e-12)
    mins = mins.to(device=DEVICE, dtype=DTYPE)
    rng = rng.to(device=DEVICE, dtype=DTYPE)
    ro = {**readout}

    idx_lookup = {name: i for i, name in enumerate(feature_names or [])}
    idx_lookup_lower = {name.lower(): i for i, name in enumerate(feature_names or [])}

    def _dim_index(key: str) -> Optional[int]:
        if key.startswith("x"):
            try:
                j = int(key[1:]) - 1
            except ValueError:
                return None
            return j if 0 <= j < len(mins) else None
        if key in idx_lookup:
            return idx_lookup[key]
        return idx_lookup_lower.get(key.lower())

    effects_out = {}
    for name, spec in (ro.get("effects", {}) or {}).items():
        spec_out = dict(spec)
        rh = spec_out.get("range_hint")
        dim_idx = None
        if isinstance(name, str):
            dim_idx = _dim_index(name)
        if dim_idx is not None and isinstance(rh, (list, tuple)) and len(rh) == 2:
            low = float(rh[0]); high = float(rh[1])
            low_n = float(((low - mins[dim_idx]) / rng[dim_idx]).clamp(1e-6, 1 - 1e-6).item())
            high_n = float(((high - mins[dim_idx]) / rng[dim_idx]).clamp(1e-6, 1 - 1e-6).item())
            spec_out["range_hint"] = [low_n, high_n]

        effects_out[name] = spec_out
    ro["effects"] = effects_out

    bumps = []
    for b in ro.get("bumps", []) or []:
        mu = b.get("mu", None)
        sigma = b.get("sigma", 0.15)
        amp = b.get("amp", 0.1)
        if mu is None:
            continue
        mu_vec = torch.tensor(mu, dtype=DTYPE, device=mins.device)
        mu_norm = ((mu_vec - mins) / rng).clamp(1e-6, 1 - 1e-6)
        if isinstance(sigma, (list, tuple)):
            sigma_vec = torch.tensor(list(sigma), dtype=DTYPE, device=mins.device)
            sigma_norm = (sigma_vec / rng).clamp_min(1e-6)
            sigma_out = [float(v) for v in sigma_norm.cpu().tolist()]
        else:
            sigma_scalar = float(sigma)
            scale = torch.mean((torch.ones_like(rng) * sigma_scalar) / rng).clamp_min(1e-6)
            sigma_out = float(scale.item())
        bumps.append({
            "mu": [float(v) for v in mu_norm.cpu().tolist()],
            "sigma": sigma_out,
            "amp": float(amp),
        })
    ro["bumps"] = bumps
    return ro

# -------------------- Prompt + LLM readout -------------------------

def llm_generate_readout(history_df: Optional[pd.DataFrame], gp_ctx: SingleTaskGP, X_obs: Tensor, Y_obs: Tensor,
                          X_pool: Tensor, sys_prompt: str,
                          temperature: float = 0.2, model: str = "gpt-4o-mini",
                          unit_to_raw_fn: Optional[Callable[[Tensor], Tensor]] = None) -> Dict[str, Any]:
    if _OPENAI_CLIENT is None:
        raise RuntimeError("LLM client is unavailable; cannot generate readout.")
    ctx = summarize_on_pool_for_llm(gp_ctx, X_obs, Y_obs, X_pool, unit_to_raw_fn=unit_to_raw_fn)
    recent = history_df.tail(30).to_dict(orient="records") if (history_df is not None and not history_df.empty) else []

    user_payload = {
        "context": ctx,
        "recent": recent,
        "instructions": "Return JSON only.",
    }
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": "Respond with JSON only:\n" + json.dumps(user_payload)},
    ]
    resp = _OPENAI_CLIENT.chat.completions.create(
        model=model, temperature=temperature, response_format={"type": "json_object"}, messages=messages,
    )

    return json.loads(resp.choices[0].message.content)


# ====================================================================
#                        CORE METHODS ON LOOKUP TABLE (single run)
# ====================================================================

def _run_random_lookup_single(lookup: LookupTable, *, n_init: int, n_iter: int, seed: int = 0,
                              init_method: str = "random") -> pd.DataFrame:
    N = lookup.n
    idxs = _select_initial_indices_lookup(lookup, n_init, seed, init_method)
    seen: set[int] = set(int(i) for i in idxs)

    if idxs:
        X_obs = lookup.X[idxs]
        Y_obs = lookup.y[idxs]
    else:
        X_obs = torch.empty((0, lookup.d), device=lookup.X.device, dtype=lookup.X.dtype)
        Y_obs = torch.empty((0,), device=lookup.y.device, dtype=lookup.y.dtype)

    rec: List[Dict[str, Any]] = []
    best_running = float('-inf')
    for i, idx in enumerate(idxs):
        y_i = float(lookup.y[idx].item())
        best_running = max(best_running, y_i)
        row = as_records_row(idx, lookup.X_raw[idx], y_i, best_running, method="random", feature_names=lookup.feature_names)
        row["iter"] = i - len(idxs)
        rec.append(row)

    g = torch.Generator(device='cpu'); g.manual_seed(int(seed) + 12345)
    best = best_running
    for t in range(n_iter):
        rem = remaining_indices(N, seen)
        if not rem: break
        pick = rem[torch.randint(low=0, high=len(rem), size=(1,), generator=g).item()]
        seen.add(pick)

        y = float(lookup.y[pick].item())
        best = max(best, y)
        row = as_records_row(pick, lookup.X_raw[pick], y, best, method="random", feature_names=lookup.feature_names)
        row["iter"] = t
        rec.append(row)

    df = pd.DataFrame(rec)
    return df

from botorch.models.transforms.outcome import Standardize # Add this import at the top

def _run_baseline_ei_lookup_single(lookup: LookupTable, *, n_init: int, n_iter: int, seed: int = 0,
                                   init_method: str = "random") -> pd.DataFrame:
    N = lookup.n

    # --- choose initial indices based on init_method ---
    idxs = _select_initial_indices_lookup(lookup, n_init, seed, init_method)

    # Build starting state & records
    seen, X_obs, Y_obs, rec = _build_initial_from_indices(lookup, idxs, method_tag="baseline_ei")

    # --- BO loop ---
    best_running = float(max([r["y"] for r in rec]) if rec else -float("inf"))
    for t in range(n_iter):
        Y_obs2 = Y_obs.unsqueeze(-1)
        gp = SingleTaskGP(X_obs, Y_obs2)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        rem = remaining_indices(N, seen)
        if not rem:
            break
        X_pool = lookup.X[rem]

        with torch.no_grad():
            train_post = gp.posterior(X_obs.unsqueeze(1))
            best_f = float(train_post.mean.max().item())
            EI = ExpectedImprovement(model=gp, best_f=best_f, maximize=True)
            ei_vals = EI(X_pool.unsqueeze(1)).reshape(-1)
            idx_local = int(torch.argmax(ei_vals))
            pick = rem[idx_local]

        seen.add(pick)
        y = float(lookup.y[pick].item())
        best_running = max(best_running, y)
        X_obs = torch.cat([X_obs, lookup.X[pick].unsqueeze(0)], dim=0)
        Y_obs = torch.cat([Y_obs, torch.tensor([y], dtype=DTYPE, device=DEVICE)], dim=0)

        row = as_records_row(pick, lookup.X_raw[pick], y, best_running,
                             method="baseline_ei", feature_names=lookup.feature_names)
        row["iter"] = t
        rec.append(row)

    return pd.DataFrame(rec)

def _run_hybrid_lookup_single(lookup: LookupTable, *, n_init: int, n_iter: int, seed: int = 0,
                              init_method: str = "random",
                              readout_source: str = "flat",  # "flat" | "llm"
                              pool_base: Optional[int] = None,
                              debug_llm: bool = False,
                              model: str = "gpt-4o-mini",
                              log_json_path: Optional[str] = None,
                              diagnose_prior: bool = False,
                              prompt_profile: str = "perfect",
                              method_tag: Optional[str] = None,
                              prior_strength: float = 1.0,
                              rho_floor: float = 0.05,
                              early_prior_boost: bool = False,
                              early_prior_steps: int = 5) -> pd.DataFrame:
    N = lookup.n

    # --- choose initial indices based on init_method ---
    idxs = _select_initial_indices_lookup(lookup, n_init, seed, init_method)

    # Build starting state & records
    method_label = method_tag or ("hybrid_llm" if readout_source == "llm" else "hybrid_flat")
    if readout_source == "llm":
        method_label = method_tag or f"hybrid_{prompt_profile}"

    seen, X_obs, Y_obs, rec = _build_initial_from_indices(lookup, idxs, method_tag=method_label)

    prior_debug: Optional[List[Dict[str, Any]]] = [] if diagnose_prior else None
    unit_to_raw_lookup = _make_unit_to_raw_fn(lookup.mins, lookup.maxs)

    # --- initial prior selection ---
    if readout_source == "llm":
        prompt_text = PROMPT_LIBRARY.get(prompt_profile)
        Y_obs2 = Y_obs.unsqueeze(-1)
        gp_ctx = SingleTaskGP(X_obs, Y_obs2)
        mll = ExactMarginalLogLikelihood(gp_ctx.likelihood, gp_ctx)
        fit_gpytorch_mll(mll)
        rem0 = remaining_indices(N, seen)
        X_pool0 = lookup.X[rem0] if rem0 else torch.empty((0, lookup.d), device=DEVICE, dtype=DTYPE)
        ro0_raw = llm_generate_readout(
            pd.DataFrame(rec),
            gp_ctx,
            X_obs,
            Y_obs2,
            X_pool0,
            sys_prompt=prompt_text,
            model=model,
            unit_to_raw_fn=unit_to_raw_lookup,
        )
        print(f"[Hybrid][lookup] seed={seed} readout ({prompt_profile}):\n"
              f"{json.dumps(ro0_raw, indent=2)}")
        ro0_sane = sanitize_readout_dim(ro0_raw, lookup.d, feature_names=lookup.feature_names)
        ro0 = _normalize_readout_to_unit_box(ro0_sane, lookup.mins, lookup.maxs, feature_names=lookup.feature_names)
        print(ro0)
    else:
        ro0 = flat_readout(feature_names=lookup.feature_names)
    prior = readout_to_prior(ro0, feature_names=lookup.feature_names)
    m0_w = 1.0

    status_log: List[Dict[str, Any]] = []
    if log_json_path is not None and readout_source == "llm":
        status_log.append({"iter": -1, "readout": ro0})

    best_running = float(max([r["y"] for r in rec]) if rec else -float("inf"))

    for t in range(n_iter):
        Y_obs2 = Y_obs.unsqueeze(-1)

        # if readout_source == "llm":
        #     gp_ctx = SingleTaskGP(X_obs, Y_obs2)
        #     mll_ctx = ExactMarginalLogLikelihood(gp_ctx.likelihood, gp_ctx)
        #     fit_gpytorch_mll(mll_ctx)
        #     rem_prompt = remaining_indices(N, seen)
        #     X_pool_prompt = lookup.X[rem_prompt] if rem_prompt else torch.empty((0, lookup.d), device=DEVICE, dtype=DTYPE)
        #     ro_t_raw = llm_generate_readout(pd.DataFrame(rec), gp_ctx, X_obs, Y_obs2, X_pool_prompt,
        #                                     sys_prompt=prompt_text, model=model)
        #     ro_t_sane = sanitize_readout_dim(ro_t_raw, lookup.d, feature_names=lookup.feature_names)
        #     ro_t = _normalize_readout_to_unit_box(ro_t_sane, lookup.mins, lookup.maxs, feature_names=lookup.feature_names)
        #     prior = readout_to_prior(ro_t)
        #     if log_json_path is not None:
        #         status_log.append({"iter": t, "readout": ro_t})

        rem = remaining_indices(N, seen)
        if not rem:
            break
        if pool_base is not None and len(rem) > pool_base:
            g = torch.Generator(device='cpu'); g.manual_seed(int(seed) + 777 + t)
            idx_sub = torch.tensor(rem)[torch.randperm(len(rem), generator=g)[:pool_base]].tolist()
            rem_eval = idx_sub
        else:
            rem_eval = rem
        X_pool = lookup.X[rem_eval]

        if early_prior_boost and t < early_prior_steps:
            prior_vals = prior.m0_torch(X_pool).reshape(-1)
            idx_local = int(torch.argmax(prior_vals))
            pick = rem_eval[idx_local]
            y = float(lookup.y[pick].item())
            best_running = max(best_running, y)
            seen.add(pick)
            X_obs = torch.cat([X_obs, lookup.X[pick].unsqueeze(0)], dim=0)
            Y_obs = torch.cat([Y_obs, torch.tensor([y], dtype=DTYPE, device=DEVICE)], dim=0)
            row = as_records_row(pick, lookup.X_raw[pick], y, best_running,
                                 method=method_label, feature_names=lookup.feature_names)
            row["iter"] = t
            row["prior_boost"] = True
            rec.append(row)
            continue

        gp_resid, alpha = fit_residual_gp(X_obs, Y_obs2, prior)
        rho = alignment_on_obs(X_obs, Y_obs2, prior)
        rho_weight = max(abs(float(rho)), rho_floor)
        m0_scale = float(alpha * m0_w * prior_strength * rho_weight)
        model_total = GPWithPriorMean(gp_resid, prior, m0_scale=m0_scale)

        with torch.no_grad():
            train_post = model_total.posterior(X_obs.unsqueeze(1))
            best_f = float(train_post.mean.max().item())
            EI = ExpectedImprovement(model=model_total, best_f=best_f, maximize=True)
            ei_vals = EI(X_pool.unsqueeze(1)).reshape(-1)
            idx_local = int(torch.argmax(ei_vals))
            pick = rem_eval[idx_local]

        if diagnose_prior:
            gp_plain = SingleTaskGP(X_obs, Y_obs2)
            mll_plain = ExactMarginalLogLikelihood(gp_plain.likelihood, gp_plain)
            fit_gpytorch_mll(mll_plain)
            with torch.no_grad():
                post_plain = gp_plain.posterior(X_obs.unsqueeze(1))
                best_plain = float(post_plain.mean.max().item())
                EI_plain = ExpectedImprovement(model=gp_plain, best_f=best_plain, maximize=True)
                ei_plain_vals = EI_plain(X_pool.unsqueeze(1)).reshape(-1)
                idx_plain_local = int(torch.argmax(ei_plain_vals))
                pick_plain = rem_eval[idx_plain_local]
                prior_vals = prior.m0_torch(X_pool).reshape(-1)
                idx_prior_local = int(torch.argmax(prior_vals))
                pick_prior = rem_eval[idx_prior_local]

            diag_entry = {
                "iter": t,
                "seed": seed,
                "hybrid_idx": int(pick),
                "baseline_idx": int(pick_plain),
                "prior_idx": int(pick_prior),
                "hybrid_equals_baseline": bool(pick == pick_plain),
                "alpha": float(alpha),
                "rho": float(rho),
                "rho_weight": float(rho_weight),
                "m0_scale": float(m0_scale),
                "hybrid_ei": float(ei_vals[idx_local].item()),
                "baseline_ei": float(ei_plain_vals[idx_plain_local].item()),
                "prior_mean_hybrid": float(prior_vals[idx_local].item()),
                "prior_mean_baseline": float(prior_vals[idx_plain_local].item()),
                "prior_mean_top": float(prior_vals[idx_prior_local].item()),
                "hybrid_raw": [float(v) for v in lookup.X_raw[pick].detach().cpu().tolist()],
                "baseline_raw": [float(v) for v in lookup.X_raw[pick_plain].detach().cpu().tolist()],
                "prior_raw": [float(v) for v in lookup.X_raw[pick_prior].detach().cpu().tolist()],
            }
            if prior_debug is not None:
                prior_debug.append(diag_entry)
            if debug_llm:
                print(f"[prior-diagnostics] seed={seed} iter={t} hybrid_idx={pick} baseline_idx={pick_plain} same={pick == pick_plain}")

        y = float(lookup.y[pick].item())
        best_running = max(best_running, y)
        seen.add(pick)
        X_obs = torch.cat([X_obs, lookup.X[pick].unsqueeze(0)], dim=0)
        Y_obs = torch.cat([Y_obs, torch.tensor([y], dtype=DTYPE, device=DEVICE)], dim=0)

        row = as_records_row(pick, lookup.X_raw[pick], y, best_running,
                             method=method_label, feature_names=lookup.feature_names)
        row["iter"] = t
        rec.append(row)

    if log_json_path is not None and readout_source == "llm":
        dirpath = os.path.dirname(log_json_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(log_json_path, 'w') as f:
            json.dump(status_log, f, indent=2)

    df = pd.DataFrame(rec)
    if diagnose_prior and prior_debug is not None:
        df.attrs["prior_debug"] = prior_debug
    return df

# ====================================================================
#                        PUBLIC METHODS (with repeats)
# ====================================================================

def run_random_lookup(lookup: LookupTable, *, n_init: int, n_iter: int, seed: int = 0,
                      repeats: int = 1, init_method: str = "random") -> pd.DataFrame:
    """Random selection over the table. If repeats>1, runs multiple seeds and concatenates results.

    init_method controls the shared initialization design (e.g., 'random', 'sobol', 'llm-si').
    """
    if repeats <= 1:
        df = _run_random_lookup_single(lookup, n_init=n_init, n_iter=n_iter, seed=seed,
                                       init_method=init_method)
        df["seed"] = seed
        return df
    dfs = []
    for r in range(repeats):
        s = seed + r
        dfr = _run_random_lookup_single(lookup, n_init=n_init, n_iter=n_iter, seed=s,
                                        init_method=init_method)
        dfr["seed"] = s
        dfs.append(dfr)
    return pd.concat(dfs, ignore_index=True)


def run_baseline_ei_lookup(lookup: LookupTable, *, n_init: int, n_iter: int, seed: int = 0,
                           repeats: int = 1, init_method: str = "random") -> pd.DataFrame:
    if repeats <= 1:
        df = _run_baseline_ei_lookup_single(lookup, n_init=n_init, n_iter=n_iter, seed=seed,
                                            init_method=init_method)
        df["seed"] = seed
        return df
    dfs = []
    for r in range(repeats):
        s = seed + r
        dfr = _run_baseline_ei_lookup_single(lookup, n_init=n_init, n_iter=n_iter, seed=s,
                                             init_method=init_method)
        dfr["seed"] = s
        dfs.append(dfr)
    return pd.concat(dfs, ignore_index=True)


def run_hybrid_lookup(lookup: LookupTable, *, n_init: int, n_iter: int, seed: int = 0,
                      repeats: int = 1, init_method: str = "random",
                      readout_source: str = "flat", pool_base: Optional[int] = None,
                      debug_llm: bool = False, model: str = "gpt-4o-mini",
                      log_json_path: Optional[str] = None,
                      diagnose_prior: bool = False,
                      prompt_profile: str = "perfect",
                      method_tag: Optional[str] = None,
                      prior_strength: float = 1.0,
                      rho_floor: float = 0.05,
                      early_prior_boost: bool = False,
                      early_prior_steps: int = 5) -> pd.DataFrame:
    if repeats <= 1:
        df = _run_hybrid_lookup_single(lookup, n_init=n_init, n_iter=n_iter, seed=seed,
                                       init_method=init_method,
                                       readout_source=readout_source, pool_base=pool_base,
                                       debug_llm=debug_llm, model=model, log_json_path=log_json_path,
                                       diagnose_prior=diagnose_prior,
                                       prompt_profile=prompt_profile,
                                       method_tag=method_tag,
                                       prior_strength=prior_strength,
                                       rho_floor=rho_floor,
                                       early_prior_boost=early_prior_boost,
                                       early_prior_steps=early_prior_steps)
        df["seed"] = seed
        if diagnose_prior and "prior_debug" in df.attrs:
            df.attrs.setdefault("prior_debug_runs", []).append({"seed": seed, "prior_debug": df.attrs["prior_debug"]})
        return df
    dfs = []
    debug_runs: List[Dict[str, Any]] = [] if diagnose_prior else []
    for r in range(repeats):
        s = seed + r
        dfr = _run_hybrid_lookup_single(lookup, n_init=n_init, n_iter=n_iter, seed=s,
                                        init_method=init_method,
                                        readout_source=readout_source, pool_base=pool_base,
                                        debug_llm=debug_llm, model=model, log_json_path=None,
                                        diagnose_prior=diagnose_prior,
                                        prompt_profile=prompt_profile,
                                        method_tag=method_tag,
                                        prior_strength=prior_strength,
                                        rho_floor=rho_floor,
                                        early_prior_boost=early_prior_boost,
                                        early_prior_steps=early_prior_steps)
        dfr["seed"] = s
        if diagnose_prior and "prior_debug" in dfr.attrs:
            debug_runs.append({"seed": s, "prior_debug": dfr.attrs["prior_debug"]})
        dfs.append(dfr)
    out = pd.concat(dfs, ignore_index=True)
    if diagnose_prior and debug_runs:
        out.attrs["prior_debug_runs"] = debug_runs
    return out


def compare_methods_from_csv(lookup: LookupTable, n_init: int = 6, n_iter: int = 25, seed: int = 0,
                             repeats: int = 1, include_hybrid: bool = True, readout_source: str = "flat",
                             init_method: str = "random", prompt_profiles = ["perfect"] ,diagnose_prior: bool = False,
                             prior_strength: float = 1.0, rho_floor: float = 0.05,
                             early_prior_boost: bool = False, early_prior_steps: int = 5) -> pd.DataFrame:
    if isinstance(prompt_profiles, str):
        prompt_profiles = [prompt_profiles]
    dfs: List[pd.DataFrame] = []
    rand = run_random_lookup(lookup, n_init=n_init, n_iter=n_iter, seed=seed,
                             repeats=repeats, init_method=init_method)
    base = run_baseline_ei_lookup(lookup, n_init=n_init, n_iter=n_iter, seed=seed,
                                  repeats=repeats, init_method=init_method)
    dfs.extend([rand, base])
    hybrid_debug: List[Dict[str, Any]] = []
    if include_hybrid:
        if readout_source == "llm":
            for profile in prompt_profiles:
                method_label = f"hybrid_{profile}"
                print(f"[Hybrid] lookup using readout '{profile}'")
                hyb = run_hybrid_lookup(lookup, n_init=n_init, n_iter=n_iter, seed=seed, repeats=repeats,
                                        init_method=init_method, readout_source="llm",
                                        diagnose_prior=diagnose_prior, prompt_profile=profile,
                                        method_tag=method_label,
                                        prior_strength=prior_strength,
                                        rho_floor=rho_floor,
                                        early_prior_boost=early_prior_boost,
                                        early_prior_steps=early_prior_steps)
                if diagnose_prior:
                    payload = hyb.attrs.get("prior_debug_runs") or hyb.attrs.get("prior_debug")
                    if payload:
                        hybrid_debug.append({"prompt": profile, "debug": payload})
                dfs.append(hyb)
        else:
            method_label = "hybrid_flat"
            print("[Hybrid] lookup using flat readout")
            hyb = run_hybrid_lookup(lookup, n_init=n_init, n_iter=n_iter, seed=seed, repeats=repeats,
                                    init_method=init_method, readout_source=readout_source,
                                    diagnose_prior=diagnose_prior, prompt_profile="perfect",
                                    method_tag=method_label,
                                    prior_strength=prior_strength,
                                    rho_floor=rho_floor,
                                    early_prior_boost=early_prior_boost,
                                    early_prior_steps=early_prior_steps)
            if diagnose_prior:
                payload = hyb.attrs.get("prior_debug_runs") or hyb.attrs.get("prior_debug")
                if payload:
                    hybrid_debug.append({"prompt": method_label, "debug": payload})
            dfs.append(hyb)

    out = pd.concat(dfs, ignore_index=True)
    if hybrid_debug:
        out.attrs["prior_debug_runs"] = hybrid_debug
    return out

# ====================================================================
#                          COMPARISON WRAPPER
# ====================================================================



# ====================================================================
#                              PLOTTING
# ====================================================================

def plot_runs_mean_lookup(hist_df: pd.DataFrame, *, methods: Optional[List[str]] = None,
                          ci: str = "sd", ax: Optional[plt.Axes] = None,
                          title: Optional[str] = None) -> plt.Axes:
    """Plot mean best-so-far vs iteration across seeds with a shaded uncertainty band.

    Args:
        hist_df: long-form DataFrame from compare_methods_from_csv (must contain columns: method, iter, best_so_far, seed).
        methods: subset of method names to plot (default: all in hist_df).
        ci: "sd" (±1 std), "sem" (±1 std/sqrt(n)), or "95ci" (±1.96*std/sqrt(n)).
        ax: optional matplotlib Axes.
        title: optional plot title.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))

    df = hist_df.copy()
    df = df[df["iter"] >= 0]  # plot only acquisition iterations

    if methods is None:
        methods = list(df["method"].unique())

    for m in methods:
        d = df[df["method"] == m]
        if d.empty:
            continue
        agg = d.groupby("iter")["best_so_far"].agg(["mean", "std", "count"]).reset_index()
        if ci == "sem":
            err = agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
        elif ci == "95ci":
            err = 1.96 * agg["std"] / np.maximum(agg["count"], 1).pow(0.5)
        else:  # "sd"
            err = agg["std"]
        x = agg["iter"].to_numpy()
        y = agg["mean"].to_numpy()
        e = err.to_numpy()
        ax.plot(x, y, label=m)
        ax.fill_between(x, y - e, y + e, alpha=0.20)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best so far")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.35)
    ax.legend()
    plt.tight_layout()
    return ax


def attach_lookup_truth(hist_df: pd.DataFrame, lookup: LookupTable) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Attach raw feature values and the lookup-table objective to each BO selection."""
    feature_cols = list(lookup.feature_names)
    lookup_df = pd.DataFrame(
        lookup.X_raw.detach().cpu().numpy(),
        columns=feature_cols
    )
    lookup_df["truth_objective"] = lookup.y.detach().cpu().numpy()
    lookup_df["idx"] = np.arange(lookup_df.shape[0])
    merged = hist_df.merge(
        lookup_df[["idx", "truth_objective", *feature_cols]],
        on="idx",
        how="left"
    )
    return merged, lookup_df, lookup.objective_name


def plot_parameter_violin(hist_truth_df: pd.DataFrame,
                          lookup_df: pd.DataFrame,
                          *,
                          methods: Optional[List[str]] = None,
                          feature_cols: Optional[List[str]] = None,
                          truth_col: str = "truth_objective",
                          objective_label: str = "Objective") -> plt.Figure:
    """Show how each method's sampled parameters distribute relative to the lookup-table optimum."""
    if methods is None:
        methods = sorted(hist_truth_df["method"].unique())
    if feature_cols is None:
        feature_cols = [c for c in lookup_df.columns if c not in {"idx", truth_col}]
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one feature.")

    best_row = lookup_df.loc[lookup_df[truth_col].idxmax()]
    fig, axes = plt.subplots(1, len(feature_cols), figsize=(5.0 * len(feature_cols), 4.8), sharey=False)
    if len(feature_cols) == 1:
        axes = [axes]

    for ax, feat in zip(axes, feature_cols):
        distributions = []
        labels = []
        for method in methods:
            vals = hist_truth_df.loc[hist_truth_df["method"] == method, feat].dropna()
            if vals.empty:
                continue
            distributions.append(vals.to_numpy())
            labels.append(method)
        if not distributions:
            ax.set_title(f"No data for {feat}")
            continue
        parts = ax.violinplot(distributions, showextrema=False)
        for body in parts["bodies"]:
            body.set_alpha(0.45)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(feat)
        ax.set_title(f"{feat} selection per method")
        ax.grid(True, alpha=0.2, axis="y")
        ax.axhline(
            float(best_row[feat]),
            color="red",
            linestyle="--",
            linewidth=1.0,
            label="Best lookup value"
        )
        overall_vals = lookup_df[feat].to_numpy()
        ax.scatter(
            np.full_like(overall_vals, fill_value=len(labels) + 0.6, dtype=float),
            overall_vals,
            s=6,
            c="gray",
            alpha=0.3,
            label="Lookup table"
        )
        ax.set_xlim(0.5, len(labels) + 1.2)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Parameter selection spread vs lookup optimum ({objective_label})", y=0.98)
    fig.tight_layout()
    return fig


def plot_parameter_violin_from_history(
    hist_df: pd.DataFrame,
    *,
    methods: Optional[List[str]] = None,
    feature_cols: Optional[List[str]] = None,
    truth_df: Optional[pd.DataFrame] = None,
    truth_label: str = "yield",
) -> plt.Figure:
    """Show how each method's samples distribute for chosen features."""
    df = hist_df.copy()
    if methods is None:
        methods = sorted(df["method"].unique())
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c.startswith("x")]
    if not feature_cols:
        raise ValueError("No feature columns (x*) present in history.")

    fig, axes = plt.subplots(
        1,
        len(feature_cols),
        figsize=(4.8 * len(feature_cols), 4.6),
        sharey=False,
    )
    if len(feature_cols) == 1:
        axes = [axes]

    best_truth = None
    if truth_df is not None and truth_label in truth_df.columns:
        best_truth = truth_df.loc[truth_df[truth_label].idxmax()]

    for ax, feat in zip(axes, feature_cols):
        violins = []
        labels = []
        for m in methods:
            vals = df.loc[df["method"] == m, feat].dropna()
            if vals.empty:
                continue
            violins.append(vals.to_numpy())
            labels.append(m)
        if not violins:
            ax.set_title(f"No data for {feat}")
            continue
        parts = ax.violinplot(violins, showmeans=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_alpha(0.45)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel(feat)
        ax.grid(True, alpha=0.25, axis="y")
        if best_truth is not None and feat in best_truth:
            ax.axhline(
                float(best_truth[feat]),
                color="red",
                linestyle="--",
                linewidth=1.0,
                label="Best empirical value",
            )
            ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Parameter focus per method", y=0.98)
    fig.tight_layout()
    return fig


def plot_method_2d_hist(
    hist_df: pd.DataFrame,
    feature_x: str,
    feature_y: str,
    *,
    methods: Optional[List[str]] = None,
    bins: int = 30,
) -> plt.Figure:
    """2D density plot showing where each method samples."""
    if methods is None:
        methods = sorted(hist_df["method"].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(5.0 * len(methods), 4.5))
    if len(methods) == 1:
        axes = [axes]
    for ax, method in zip(axes, methods):
        subset = hist_df[hist_df["method"] == method]
        if subset.empty:
            ax.set_title(f"No data for {method}")
            continue
        h = ax.hist2d(
            subset[feature_x],
            subset[feature_y],
            bins=bins,
            cmap="viridis",
        )
        fig.colorbar(h[3], ax=ax)
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        ax.set_title(method)
        ax.grid(True, alpha=0.15)
    fig.suptitle(f"{feature_x} vs {feature_y} sampling density", y=0.98)
    fig.tight_layout()
    return fig
#%%
if __name__ == "__main__":
    domain = build_continuous_domain()
    summary = domain.metadata or {}
    print("UGI continuous oracle summary:")
    print(f" - Training rows: {summary.get('n_training_rows', 'N/A')}")
    metrics = summary.get("oracle_metrics") or {}
    if metrics:
        print(
            " - RF metrics: "
            f"OOB={metrics.get('oob_score', float('nan')):.3f}, "
            f"RMSE={metrics.get('rmse', float('nan')):.5f}, "
            f"R^2={metrics.get('r2', float('nan')):.3f}"
        )

    hist = compare_methods_continuous(
        domain,
        n_init=5,
        n_iter=20,
        seed=41,
        repeats=3,
        include_hybrid=True,
        readout_source="llm",
        prompt_profiles="best",
        early_prior_boost=True,
        early_prior_steps=5,
        diagnose_prior=True,
    )
    plot_runs_mean_lookup(hist)


#%%
    truth_df = pd.read_csv("ugi_merged_dataset.csv")

    feature_subset = domain.feature_names[:2]
    methods_to_show = [m for m in ["random", "baseline_ei"] if m in hist["method"].unique()]
    if "hybrid_bad" in hist["method"].unique():
        methods_to_show.append("hybrid_bad")
    if "hybrid_best" in hist["method"].unique():
        methods_to_show.append("hybrid_best")    
    if methods_to_show and feature_subset:
        plot_parameter_violin_from_history(
            hist,
            methods=methods_to_show,
            feature_cols=[f"x{domain.feature_names.index(name)+1}" for name in feature_subset],
            truth_df=truth_df,
        )
        plot_method_2d_hist(
            hist,
            feature_x=f"x{domain.feature_names.index(feature_subset[0])+1}",
            feature_y=f"x{domain.feature_names.index(feature_subset[1])+1}",
            methods=methods_to_show,
        )
#%%

# debug_runs = hist.attrs.get("prior_debug_runs", [])
