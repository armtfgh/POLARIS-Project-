#!/usr/bin/env python3
"""
prompt_generator.py
------------------
Generate a UGI readout JSON from a prompt using OpenAI, then score the induced
prior mean m0(x) against dataset truth (correlation + top-k metrics).

This file intentionally does NOT contain any heuristic prompt-to-readout logic.

Example (Jupyter)
-----------------
```python
from bo_readout_prompts import SYS_PROMPTS_BEST, SYS_PROMPTS_BAD
from prompt_generator import score_prompt_via_llm, score_prompt_library_via_llm, print_summary

best = score_prompt_via_llm(prompt=SYS_PROMPTS_BEST, csv_path="ugi_merged_dataset.csv")
bad  = score_prompt_via_llm(prompt=SYS_PROMPTS_BAD,  csv_path="ugi_merged_dataset.csv")
print_summary([{"name": "best", **best}, {"name": "bad", **bad}])

rows = score_prompt_library_via_llm(csv_path="ugi_merged_dataset.csv")
print_summary(rows)
```
"""

#%%
from __future__ import annotations

from dataclasses import dataclass
import json
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch import Tensor

# prior_gp calls torch.cuda.is_available() at import time; suppress noisy warnings in CPU-only envs.
warnings.filterwarnings("ignore", message=r"CUDA initialization:.*")

from prior_gp import Prior, alignment_on_obs
from readout_schema import readout_to_prior

DEVICE = torch.device("cpu")
DTYPE = torch.float32


@dataclass
class LookupTable:
    X_raw: Tensor  # (N, d) raw features
    y: Tensor  # (N,) objective
    X: Tensor  # (N, d) normalized to [0,1]
    mins: Tensor  # (d,)
    maxs: Tensor  # (d,)
    feature_names: List[str]
    objective_name: str

    @property
    def n(self) -> int:
        return int(self.X.shape[0])

    @property
    def d(self) -> int:
        return int(self.X.shape[1])


def load_lookup_csv(
    path: str,
    *,
    device: torch.device = DEVICE,
    dtype: torch.dtype = DTYPE,
    objective_col: str = "yield",
    impute_features: Optional[str] = "median",  # None | 'median'
    aggregate_duplicates: Optional[str] = "mean",  # None | 'mean' | 'median' | 'max'
) -> LookupTable:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least 2 columns (>=1 feature + 1 objective)")

    cols = list(df.columns)
    if objective_col not in cols:
        raise ValueError(f"objective_col={objective_col!r} not found in CSV columns={cols!r}")
    feat_cols = [c for c in cols if c != objective_col]

    for c in feat_cols + [objective_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    df[objective_col] = df[objective_col].replace([np.inf, -np.inf], np.nan)

    mask_y = df[objective_col].notna()
    if impute_features == "median":
        med = df.loc[mask_y, feat_cols].median(numeric_only=True)
        df.loc[:, feat_cols] = df[feat_cols].fillna(med)
        mask_x = df[feat_cols].notna().all(axis=1)
    else:
        mask_x = df[feat_cols].notna().all(axis=1)

    df_clean = df.loc[mask_y & mask_x].reset_index(drop=True)
    if df_clean.shape[0] < 2:
        raise ValueError("After cleaning, fewer than 2 valid rows remain.")

    if aggregate_duplicates:
        agg = aggregate_duplicates.lower()
        if agg not in {"mean", "median", "max"}:
            raise ValueError("aggregate_duplicates must be one of {'mean','median','max', None}")
        if df_clean.duplicated(subset=feat_cols, keep=False).any():
            if agg == "mean":
                df_clean = df_clean.groupby(feat_cols, as_index=False)[objective_col].mean()
            elif agg == "median":
                df_clean = df_clean.groupby(feat_cols, as_index=False)[objective_col].median()
            else:
                df_clean = df_clean.groupby(feat_cols, as_index=False)[objective_col].max()

    X_raw_np = df_clean[feat_cols].to_numpy(dtype=np.float64)
    y_np = df_clean[objective_col].to_numpy(dtype=np.float64)

    X_raw = torch.tensor(X_raw_np, dtype=dtype, device=device)
    y = torch.tensor(y_np, dtype=dtype, device=device).reshape(-1)

    mins = X_raw.min(dim=0).values
    maxs = X_raw.max(dim=0).values
    rng = (maxs - mins).clamp_min(1e-12)
    X = (X_raw - mins) / rng

    return LookupTable(
        X_raw=X_raw,
        y=y,
        X=X,
        mins=mins,
        maxs=maxs,
        feature_names=[str(c) for c in feat_cols],
        objective_name=str(objective_col),
    )


def _sanitize_readout_minimal(ro: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(ro or {})
    out["effects"] = out.get("effects") or {}
    out["interactions"] = out.get("interactions") or []
    out["bumps"] = out.get("bumps") or []
    out["constraints"] = out.get("constraints") or []
    if not isinstance(out["effects"], dict):
        out["effects"] = {}
    if not isinstance(out["interactions"], list):
        out["interactions"] = []
    if not isinstance(out["bumps"], list):
        out["bumps"] = []
    if not isinstance(out["constraints"], list):
        out["constraints"] = []
    return out


def normalize_readout_to_unit_box(
    readout: Dict[str, Any],
    mins: Tensor,
    maxs: Tensor,
    *,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convert raw-scale readout (range_hint/mu/sigma in original units) to unit-box coordinates."""
    if readout is None:
        return {"effects": {}, "interactions": [], "bumps": [], "constraints": []}
    rng = (maxs - mins).clamp_min(1e-12)
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

    effects_out: Dict[str, Any] = {}
    for name, spec in (ro.get("effects", {}) or {}).items():
        spec_out = dict(spec or {})
        rh = spec_out.get("range_hint")
        dim_idx = _dim_index(str(name)) if isinstance(name, str) else None
        if dim_idx is not None and isinstance(rh, (list, tuple)) and len(rh) == 2:
            low = float(rh[0])
            high = float(rh[1])
            low_n = float(((low - mins[dim_idx]) / rng[dim_idx]).clamp(1e-6, 1 - 1e-6).item())
            high_n = float(((high - mins[dim_idx]) / rng[dim_idx]).clamp(1e-6, 1 - 1e-6).item())
            spec_out["range_hint"] = [low_n, high_n]
        effects_out[str(name)] = spec_out
    ro["effects"] = effects_out

    bumps: List[Dict[str, Any]] = []
    for b in ro.get("bumps", []) or []:
        mu = (b or {}).get("mu", None)
        sigma = (b or {}).get("sigma", 0.15)
        amp = (b or {}).get("amp", 0.1)
        if mu is None:
            continue

        mu_vec = torch.tensor(list(mu), dtype=DTYPE, device=mins.device)
        mu_norm = ((mu_vec - mins) / rng).clamp(1e-6, 1 - 1e-6)

        if isinstance(sigma, (list, tuple)):
            sigma_vec = torch.tensor(list(sigma), dtype=DTYPE, device=mins.device)
            sigma_norm = (sigma_vec / rng).clamp_min(1e-6)
            sigma_out: Any = [float(v) for v in sigma_norm.detach().cpu().tolist()]
        else:
            sigma_scalar = float(sigma)
            scale = torch.mean((torch.ones_like(rng) * sigma_scalar) / rng).clamp_min(1e-6)
            sigma_out = float(scale.item())

        bumps.append(
            {"mu": [float(v) for v in mu_norm.detach().cpu().tolist()], "sigma": sigma_out, "amp": float(amp)}
        )
    ro["bumps"] = bumps

    # constraints: per-dimension forbidden intervals
    constraints_out: List[Dict[str, Any]] = []
    for c in ro.get("constraints", []) or []:
        if not isinstance(c, dict):
            continue
        var = c.get("var", None)
        r = c.get("range", None)
        if var is None or not isinstance(r, (list, tuple)) or len(r) != 2:
            continue
        dim_idx = _dim_index(str(var))
        if dim_idx is None:
            continue

        low = float(r[0])
        high = float(r[1])
        low_n = float(((low - mins[dim_idx]) / rng[dim_idx]).clamp(1e-6, 1 - 1e-6).item())
        high_n = float(((high - mins[dim_idx]) / rng[dim_idx]).clamp(1e-6, 1 - 1e-6).item())
        if high_n < low_n:
            low_n, high_n = high_n, low_n

        c_out = dict(c)
        c_out["var"] = str(var)
        c_out["range"] = [float(low_n), float(high_n)]
        constraints_out.append(c_out)
    ro["constraints"] = constraints_out
    return ro


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    sa = pd.Series(a).rank(method="average").to_numpy(dtype=np.float64)
    sb = pd.Series(b).rank(method="average").to_numpy(dtype=np.float64)
    sa = sa - sa.mean()
    sb = sb - sb.mean()
    den = float(np.sqrt(np.dot(sa, sa) * np.dot(sb, sb)) + 1e-12)
    return float(np.dot(sa, sb) / den) if den > 0 else 0.0


def _ndcg_at_k(pred_scores: np.ndarray, true_relevance: np.ndarray, *, k: int) -> float:
    """Compute NDCG@k treating `true_relevance` as graded relevance."""
    n = int(true_relevance.shape[0])
    if n <= 0:
        return 0.0
    k = int(max(1, min(k, n)))

    rel = true_relevance.astype(np.float64).reshape(-1)
    rel = rel - float(np.min(rel))
    if float(np.max(rel)) <= 0.0:
        return 0.0

    order_pred = np.argsort(pred_scores, kind="mergesort")[::-1][:k]
    order_ideal = np.argsort(rel, kind="mergesort")[::-1][:k]

    discounts = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = float(np.sum(rel[order_pred] * discounts))
    idcg = float(np.sum(rel[order_ideal] * discounts))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _tail_spearman(m0: np.ndarray, y: np.ndarray, *, q: float = 0.8) -> float:
    """Spearman correlation restricted to the top (1-q) quantile of y."""
    y = y.reshape(-1).astype(np.float64)
    m0 = m0.reshape(-1).astype(np.float64)
    if y.size < 3:
        return 0.0
    thr = float(np.quantile(y, q))
    idx = np.where(y >= thr)[0]
    if idx.size < 3:
        return 0.0
    return _spearman_corr(m0[idx], y[idx])


def score_prior_against_truth(
    prior: Prior,
    X_unit: Tensor,
    y: Tensor,
    *,
    topk_frac: float = 0.01,
    tail_q: float = 0.8,
) -> Dict[str, float]:
    with torch.no_grad():
        m0 = prior.m0_torch(X_unit).reshape(-1)
    pearson = float(alignment_on_obs(X_unit, y, prior))

    m0_np = m0.detach().cpu().numpy().astype(np.float64)
    y_np = y.detach().cpu().numpy().reshape(-1).astype(np.float64)
    spearman = _spearman_corr(m0_np, y_np)

    n = int(len(y_np))
    k = int(max(1, round(topk_frac * n)))
    rng = np.random.default_rng(0)  # deterministic tie-breaker
    m0_j = m0_np + 1e-12 * rng.standard_normal(size=m0_np.shape)
    y_j = y_np + 1e-12 * rng.standard_normal(size=y_np.shape)
    top_m0 = set(np.argpartition(m0_j, -k)[-k:].tolist())
    top_y = set(np.argpartition(y_j, -k)[-k:].tolist())
    overlap = float(len(top_m0 & top_y) / float(k))
    topk_mean_y = float(np.mean(y_np[list(top_m0)])) if top_m0 else float("nan")

    ndcg_k = _ndcg_at_k(m0_np, y_np, k=k)
    tail_spearman = _tail_spearman(m0_np, y_np, q=tail_q)
    idx_prior_max = int(np.argmax(m0_np)) if n > 0 else -1
    y_at_prior_max = float(y_np[idx_prior_max]) if idx_prior_max >= 0 else float("nan")
    y_max = float(np.max(y_np)) if n > 0 else float("nan")
    y_min = float(np.min(y_np)) if n > 0 else float("nan")
    regret = float(y_max - y_at_prior_max) if n > 0 else float("nan")
    denom = float(max(y_max - y_min, 1e-12)) if n > 0 else float("nan")
    regret_norm = float(regret / denom) if n > 0 else float("nan")

    return {
        "pearson_alignment": pearson,
        "spearman_alignment": float(spearman),
        "topk_frac": float(topk_frac),
        "topk_k": float(k),
        "topk_overlap": overlap,
        "topk_mean_y": topk_mean_y,
        "tail_q": float(tail_q),
        "tail_spearman": float(tail_spearman),
        "ndcg_at_topk": float(ndcg_k),
        "prior_argmax_y": float(y_at_prior_max),
        "true_max_y": float(y_max),
        "simple_regret": float(regret),
        "simple_regret_norm": float(regret_norm),
    }


def openai_generate_readout(
    *,
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 800,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Call OpenAI to generate a JSON readout.

    The `prompt` should already include the UGI dataset context + response schema instructions.
    """
    try:
        import httpx  # type: ignore
        from openai import OpenAI  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("LLM mode requires `openai` and `httpx` to be installed.") from exc

    client = OpenAI(api_key=api_key, http_client=httpx.Client(verify=False))
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "Return STRICT JSON only (no prose)."},
    ]

    # Prefer chat.completions with JSON response_format; fall back if needed.
    raw_text: Optional[str] = None
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=messages,
        )
        raw_text = resp.choices[0].message.content
    except Exception:
        resp = client.responses.create(
            model=model,
            input=messages,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        # Best-effort extraction
        raw_text = getattr(resp, "output_text", None) or str(resp)

    if not raw_text:
        raise RuntimeError("OpenAI returned empty response.")

    try:
        ro = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model did not return valid JSON. Raw text:\n{raw_text}") from exc

    return _sanitize_readout_minimal(ro)


def score_prompt_via_llm(
    *,
    prompt: str,
    csv_path: str = "ugi_merged_dataset.csv",
    objective_col: str = "yield",
    topk_frac: float = 0.01,
    tail_q: float = 0.8,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    lookup = load_lookup_csv(csv_path, objective_col=objective_col, device=DEVICE, dtype=DTYPE)

    ro_raw = openai_generate_readout(prompt=prompt, model=model, api_key=api_key)
    ro_unit = normalize_readout_to_unit_box(ro_raw, lookup.mins, lookup.maxs, feature_names=lookup.feature_names)
    prior = readout_to_prior(ro_unit, feature_names=lookup.feature_names)
    metrics = score_prior_against_truth(prior, lookup.X, lookup.y, topk_frac=topk_frac, tail_q=tail_q)

    return {
        "metrics": metrics,
        "readout_raw": ro_raw,
        "readout_unit": ro_unit,
        "n": lookup.n,
        "d": lookup.d,
        "objective": lookup.objective_name,
        "features": lookup.feature_names,
    }


def score_prompt_library_via_llm(
    *,
    csv_path: str = "ugi_merged_dataset.csv",
    objective_col: str = "yield",
    topk_frac: float = 0.01,
    tail_q: float = 0.8,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    from bo_readout_prompts import (
        SYS_PROMPTS_BAD,
        SYS_PROMPTS_BEST,
        SYS_PROMPTS_GOOD,
        SYS_PROMPTS_MEDIUM,
        SYS_PROMPTS_PERFECT,
        SYS_PROMPTS_RANDOM,
        SYS_PROMPT_MINIMAL_HUMAN,
        SYS_PROMPTS_CUSTOM,
    )

    prompt_library = {
        "perfect": SYS_PROMPTS_PERFECT,
        "good": SYS_PROMPTS_GOOD,
        "medium": SYS_PROMPTS_MEDIUM,
        "minimal": SYS_PROMPT_MINIMAL_HUMAN,
        "random": SYS_PROMPTS_RANDOM,
        "bad": SYS_PROMPTS_BAD,
        "custom": SYS_PROMPTS_CUSTOM,
        "best": SYS_PROMPTS_BEST,
    }

    out: List[Dict[str, Any]] = []
    for name, prompt in prompt_library.items():
        scored = score_prompt_via_llm(
            prompt=prompt,
            csv_path=csv_path,
            objective_col=objective_col,
            topk_frac=topk_frac,
            tail_q=tail_q,
            model=model,
            api_key=api_key,
        )
        out.append({"name": name, **scored})
    return out


def print_summary(rows: List[Dict[str, Any]], *, sort_by: str = "ndcg_at_topk") -> None:
    rows = sorted(rows, key=lambda r: float(r["metrics"].get(sort_by, float("-inf"))), reverse=True)
    for r in rows:
        m = r["metrics"]
        print(
            f"{r['name']:>10s}"
            f" | ndcg@topk={m.get('ndcg_at_topk', float('nan')):.3f}"
            f" | overlap={m.get('topk_overlap', float('nan')):.3f}"
            f" | tailρ={m.get('tail_spearman', float('nan')):+.3f}"
            f" | regret={m.get('simple_regret', float('nan')):.4f}"
        )


__all__ = [
    "openai_generate_readout",
    "normalize_readout_to_unit_box",
    "score_prior_against_truth",
    "score_prompt_via_llm",
    "score_prompt_library_via_llm",
    "print_summary",
    "run_demo",
]
#%%
from os import getenv
api_key = getenv("OPENAI_API_KEY", None)
#%%
def run_demo() -> Dict[str, Any]:
    """Notebook-friendly demo: scores BEST vs BAD and the full prompt library."""
    from os import getenv
    from bo_readout_prompts import SYS_PROMPTS_BEST, SYS_PROMPTS_BAD

    api_key = getenv("OPENAI_API_KEY", None)

    best = score_prompt_via_llm(
        prompt=SYS_PROMPTS_BEST,
        csv_path="ugi_merged_dataset.csv",
        objective_col="yield",
        topk_frac=0.01,
        tail_q=0.8,
        model="gpt-4o-mini",
        api_key=api_key,
    )
    bad = score_prompt_via_llm(
        prompt=SYS_PROMPTS_BAD,
        csv_path="ugi_merged_dataset.csv",
        objective_col="yield",
        topk_frac=0.01,
        tail_q=0.8,
        model="gpt-4o-mini",
        api_key=api_key,
    )

    print("BEST vs BAD:")
    print_summary([{"name": "best", **best}, {"name": "bad", **bad}])

    # rows = score_prompt_library_via_llm(
    #     csv_path="ugi_merged_dataset.csv",
    #     objective_col="yield",
    #     topk_frac=0.01,
    #     tail_q=0.8,
    #     model="gpt-4o-mini",
    #     api_key=api_key,
    # )
    # print("\nFull prompt library:")
    # print_summary(rows)

    # return {"best": best, "bad": bad, "rows": rows}
#%%
if __name__ == "__main__":
    # Run this file directly (or in Jupyter: `%run prompt_generator.py`) to execute the demo.
    _ = run_demo()

# %%
