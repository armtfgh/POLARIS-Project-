#!/usr/bin/env python3
"""
prompt_generator.py
------------------
Deterministically (no network / no LLM) converts a natural-language prompt into a
UGI-style readout JSON, builds the corresponding prior mean m0(x), and scores how
well that m0 aligns with ground-truth data.

Intended use right now:
- Score the prompt templates in `bo_readout_prompts.py` (e.g. BEST vs BAD) against
  the available UGI dataset CSV.

Notes
-----
- This is a heuristic parser. It is not meant to replicate an LLM; it just gives
  a consistent way to turn prompt text into a readout so we can sanity-check
  whether "best" prompts score better than "bad" prompts under simple metrics.
- Alignment metrics:
  * `pearson_alignment`: Pearson corr(m0(X), y(X))
  * `spearman_alignment`: Pearson corr(rank(m0(X)), rank(y(X)))
  * `topk_overlap`: overlap between top-k sets of m0 and y
  * `topk_mean_y`: mean(y) among prior top-k points
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import re
import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

# prior_gp calls torch.cuda.is_available() at import time; suppress noisy warnings in CPU-only envs.
warnings.filterwarnings("ignore", message=r"CUDA initialization:.*")

from prior_gp import Prior, alignment_on_obs
from readout_schema import readout_to_prior, flat_readout


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
    impute_features: Optional[str] = "median",  # None | 'median'
    aggregate_duplicates: Optional[str] = "mean",  # None | 'mean' | 'median' | 'max'
    objective_col: Optional[str] = None,
) -> LookupTable:
    """Load a CSV as a lookup table (features + objective).

    By default, uses `objective_col` if provided; otherwise prefers a column named
    'yield' if present; otherwise falls back to the last column.
    """
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("CSV must have at least 2 columns (>=1 feature + 1 objective)")

    cols = list(df.columns)
    if objective_col is None:
        if "yield" in cols:
            obj_col = "yield"
        else:
            obj_col = cols[-1]
    else:
        if objective_col not in cols:
            raise ValueError(f"objective_col={objective_col!r} not found in CSV columns.")
        obj_col = objective_col

    feat_cols = [c for c in cols if c != obj_col]

    for c in feat_cols + [obj_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    df[obj_col] = df[obj_col].replace([np.inf, -np.inf], np.nan)

    mask_y = df[obj_col].notna()
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
                df_clean = df_clean.groupby(feat_cols, as_index=False)[obj_col].mean()
            elif agg == "median":
                df_clean = df_clean.groupby(feat_cols, as_index=False)[obj_col].median()
            else:
                df_clean = df_clean.groupby(feat_cols, as_index=False)[obj_col].max()

    X_raw_np = df_clean[feat_cols].to_numpy(dtype=np.float64)
    y_np = df_clean[obj_col].to_numpy(dtype=np.float64)

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
        objective_name=str(obj_col),
    )


def normalize_readout_to_unit_box(
    readout: Dict[str, Any],
    mins: Tensor,
    maxs: Tensor,
    *,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convert raw-scale readout (range_hint/mu/sigma in original units) to unit-box coordinates."""
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

    effects_out: Dict[str, Any] = {}
    for name, spec in (ro.get("effects", {}) or {}).items():
        spec_out = dict(spec)
        rh = spec_out.get("range_hint")
        dim_idx = None
        if isinstance(name, str):
            dim_idx = _dim_index(name)
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
            sigma_out: Any = [float(v) for v in sigma_norm.cpu().tolist()]
        else:
            sigma_scalar = float(sigma)
            scale = torch.mean((torch.ones_like(rng) * sigma_scalar) / rng).clamp_min(1e-6)
            sigma_out = float(scale.item())
        bumps.append({"mu": [float(v) for v in mu_norm.cpu().tolist()], "sigma": sigma_out, "amp": float(amp)})
    ro["bumps"] = bumps
    return ro


_EFFECT_CANON = {
    "increase": "increasing",
    "increasing": "increasing",
    "decrease": "decreasing",
    "decreasing": "decreasing",
    "flat": "flat",
    "peak": "nonmonotone-peak",
    "peaked": "nonmonotone-peak",
    "nonmonotone-peak": "nonmonotone-peak",
    "valley": "nonmonotone-valley",
    "nonmonotone-valley": "nonmonotone-valley",
}


def _prompt_body(prompt: str) -> str:
    """Strip the boilerplate response format section to reduce parsing noise."""
    p = prompt or ""
    # bo_readout_prompts.py includes "RESPONSE FORMAT" before the schema.
    for token in ("RESPONSE FORMAT", "RESPONSE_FORMAT"):
        i = p.find(token)
        if i != -1:
            p = p[:i]
            break

    # If this is one of the bundled prompt templates, drop the shared dataset context
    # so scores are driven by the body differences (best vs bad, etc.).
    low = p.lower()
    j = low.rfind("yield vs x4")
    if j != -1:
        k = p.find("\n", j)
        if k != -1:
            return p[k + 1 :]
    return p


def _first_float(text: str) -> Optional[float]:
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return float(m.group(0)) if m else None


def _extract_bracket_floats(text: str) -> Optional[List[float]]:
    # Accept "mu=[...]" or "mu≈[...]" forms
    m = re.search(r"\[\s*([-+0-9eE\.,\s]+)\s*\]", text)
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(",")]
    out: List[float] = []
    for p in parts:
        if not p:
            continue
        try:
            out.append(float(p))
        except ValueError:
            return None
    return out if out else None


def prompt_to_readout(
    prompt: str,
    *,
    mins: Optional[Sequence[float]] = None,
    maxs: Optional[Sequence[float]] = None,
    d: int = 4,
) -> Dict[str, Any]:
    """
    Heuristically convert prompt text into a readout JSON (raw scale).
    The returned readout is compatible with `normalize_readout_to_unit_box`.
    """
    body = _prompt_body(prompt)
    text = body.lower()

    def _default_range(i: int) -> List[float]:
        if mins is not None and maxs is not None and i < len(mins) and i < len(maxs):
            return [float(mins[i]), float(maxs[i])]
        return [0.0, 1.0]

    effects: Dict[str, Dict[str, Any]] = {}
    for i in range(d):
        key = f"x{i+1}"
        eff = "flat"
        # direct keywords
        if re.search(rf"\b{key}\b[^\n]*?\b(nonmonotone-peak|nonmonotone-valley|increasing|decreasing|flat)\b", text):
            m = re.search(
                rf"\b{key}\b[^\n]*?\b(nonmonotone-peak|nonmonotone-valley|increasing|decreasing|flat)\b",
                text,
            )
            if m:
                eff = m.group(1)
        else:
            # indirect phrasing
            if re.search(rf"\b{key}\b[^\n]*\bmaximi", text) or re.search(rf"\b{key}\b[^\n]*\bshould be maxim", text):
                eff = "increasing"
            elif re.search(rf"\b{key}\b[^\n]*\bminimi", text) or re.search(rf"\b{key}\b[^\n]*\bkeep at the minimum", text):
                eff = "decreasing"
            elif re.search(rf"\b{key}\b[^\n]*\b(peaked?|peak)\b", text):
                eff = "nonmonotone-peak"

        eff = _EFFECT_CANON.get(eff, "flat")

        # scale/confidence heuristics
        scale = 0.5
        conf = 0.6
        # numeric hints like "scale~0.3" or "scale≈1.0"
        m_scale = re.search(rf"\b{key}\b[^\n]*scale\s*[~≈=]\s*([-+]?\d*\.?\d+)", text)
        if m_scale:
            scale = float(m_scale.group(1))
        if re.search(rf"\b{key}\b[^\n]*(strong|extremely tight|force)", text):
            scale, conf = max(scale, 0.85), max(conf, 0.85)
        elif re.search(rf"\b{key}\b[^\n]*(mild|gentle|weak|cautious)", text):
            scale, conf = min(scale, 0.35), min(conf, 0.5)
        elif re.search(rf"\b{key}\b[^\n]*(moderate|supportive)", text):
            scale, conf = min(max(scale, 0.55), 0.75), min(max(conf, 0.55), 0.8)

        scale = float(np.clip(scale, 0.0, 1.0))
        conf = float(np.clip(conf, 0.0, 1.0))

        # range hints like "x1 in [118, 125]" (raw scale)
        range_hint = None
        m_rng = re.search(
            rf"\b{key}\b\s*in\s*\[\s*([-+0-9eE\.]+)\s*,\s*([-+0-9eE\.]+)\s*\]",
            body,
            flags=re.IGNORECASE,
        )
        if m_rng:
            lo = float(m_rng.group(1))
            hi = float(m_rng.group(2))
            range_hint = [lo, hi] if lo <= hi else [hi, lo]
        else:
            # "centered at 0.120" => synthesize a narrow window if sigma is present
            m_center = re.search(rf"\b{key}\b[^\n]*\bcenter(?:ed|ed)\s*at\s*([-+0-9eE\.]+)", text)
            if not m_center:
                m_center = re.search(rf"\b{key}\b[^\n]*\bcent(?:re|er)\s*at\s*([-+0-9eE\.]+)", text)
            if m_center:
                center = float(m_center.group(1))
                m_sig = re.search(rf"\b{key}\b[^\n]*\bsigma\s*[~≈=]\s*([-+0-9eE\.]+)", text)
                width = 0.08 * (maxs[i] - mins[i]) if (mins is not None and maxs is not None and i < len(mins) and i < len(maxs)) else 0.1
                if m_sig:
                    width = 4.0 * float(m_sig.group(1))
                range_hint = [center - 0.5 * width, center + 0.5 * width]

        if range_hint is None:
            range_hint = _default_range(i)

        effects[key] = {
            "effect": eff,
            "scale": scale,
            "confidence": conf,
            "range_hint": [float(range_hint[0]), float(range_hint[1])],
        }

    # interactions: parse "(x2, x3) synergy" / "(x1,x3) tradeoff" patterns
    interactions: List[Dict[str, Any]] = []
    for m in re.finditer(r"\(\s*(x\d)\s*,\s*(x\d)\s*\)\s*(synergy|tradeoff|antagonism)", text):
        a, b, t = m.group(1), m.group(2), m.group(3)
        typ = "synergy" if t == "synergy" else "tradeoff"
        interactions.append({"vars": [a, b], "type": typ, "note": ""})

    # bumps: parse "mu=[...], sigma=[...], amp=..." (raw scale)
    bumps: List[Dict[str, Any]] = []
    # Accept multiple bumps; grab all mu occurrences and then try to find sigma/amp in the same line.
    for line in body.splitlines():
        if "mu" not in line.lower():
            continue
        if not re.search(r"\bmu\b", line, flags=re.IGNORECASE):
            continue
        mu_vals = None
        if re.search(r"\bmu\s*[=≈]\s*\[", line, flags=re.IGNORECASE):
            mu_vals = _extract_bracket_floats(line)
        if not mu_vals:
            continue

        sigma_vals: Any = 0.15
        if re.search(r"\bsigma\s*[=≈]\s*\[", line, flags=re.IGNORECASE):
            sv = _extract_bracket_floats(re.sub(r"^.*sigma\s*[=≈]\s*", "", line, flags=re.IGNORECASE))
            if sv:
                sigma_vals = sv
        else:
            # allow scalar "sigma ~ 0.03"
            m_sig = re.search(r"\bsigma\s*[~≈=]\s*([-+0-9eE\.]+)", line, flags=re.IGNORECASE)
            if m_sig:
                sigma_vals = float(m_sig.group(1))

        amp = 0.1
        m_amp = re.search(r"\bamp\s*[=≈]\s*([-+0-9eE\.]+)", line, flags=re.IGNORECASE)
        if m_amp:
            raw = m_amp.group(1).strip().rstrip(".,;")
            amp = float(raw) if raw else amp

        bumps.append({"mu": [float(v) for v in mu_vals], "sigma": sigma_vals, "amp": float(amp)})

    # Fallback: prompts like SYS_PROMPTS_BEST describe a bump "at [..]" without "mu=".
    if not bumps:
        for line in body.splitlines():
            if "bump" not in line.lower():
                continue
            vals = _extract_bracket_floats(line)
            if not vals or len(vals) < d:
                continue
            mu_vals = vals[:d]
            # If sigma isn't explicitly given, derive from the effect range hints (if tight).
            sigma_vals: Any = 0.15
            m_sig = re.search(r"\bsigma\s*[~≈=]\s*([-+0-9eE\.]+)", line, flags=re.IGNORECASE)
            if m_sig:
                raw = m_sig.group(1).strip().rstrip(".,;")
                sigma_vals = float(raw)
            else:
                sigmas: List[float] = []
                for i in range(d):
                    rh = effects.get(f"x{i+1}", {}).get("range_hint")
                    if isinstance(rh, (list, tuple)) and len(rh) == 2:
                        lo, hi = float(rh[0]), float(rh[1])
                        sigmas.append(max(abs(hi - lo) / 3.0, 1e-6))
                    else:
                        sigmas.append(0.15)
                sigma_vals = sigmas
            bumps.append({"mu": [float(v) for v in mu_vals], "sigma": sigma_vals, "amp": 0.15})
            break

    return {"effects": effects, "interactions": interactions, "bumps": bumps}


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    sa = pd.Series(a).rank(method="average").to_numpy(dtype=np.float64)
    sb = pd.Series(b).rank(method="average").to_numpy(dtype=np.float64)
    sa = sa - sa.mean()
    sb = sb - sb.mean()
    den = float(np.sqrt(np.dot(sa, sa) * np.dot(sb, sb)) + 1e-12)
    return float(np.dot(sa, sb) / den) if den > 0 else 0.0


def score_prior_against_truth(
    prior: Prior,
    X_unit: Tensor,
    y: Tensor,
    *,
    topk_frac: float = 0.01,
) -> Dict[str, float]:
    """Compute alignment and a couple rank/top-k sanity metrics."""
    with torch.no_grad():
        m0 = prior.m0_torch(X_unit).reshape(-1)
    pearson = float(alignment_on_obs(X_unit, y, prior))

    m0_np = m0.detach().cpu().numpy().astype(np.float64)
    y_np = y.detach().cpu().numpy().reshape(-1).astype(np.float64)

    spearman = _spearman_corr(m0_np, y_np)
    n = int(len(y_np))
    k = int(max(1, round(topk_frac * n)))
    # Break ties deterministically; otherwise flat priors can spuriously "match" dataset ordering.
    rng = np.random.default_rng(0)
    m0_j = m0_np + 1e-12 * rng.standard_normal(size=m0_np.shape)
    y_j = y_np + 1e-12 * rng.standard_normal(size=y_np.shape)
    top_m0 = set(np.argpartition(m0_j, -k)[-k:].tolist())
    top_y = set(np.argpartition(y_j, -k)[-k:].tolist())
    overlap = float(len(top_m0 & top_y) / float(k))
    topk_mean_y = float(np.mean(y_np[list(top_m0)])) if top_m0 else float("nan")

    return {
        "pearson_alignment": pearson,
        "spearman_alignment": float(spearman),
        "topk_frac": float(topk_frac),
        "topk_k": float(k),
        "topk_overlap": overlap,
        "topk_mean_y": topk_mean_y,
    }


def score_prompt_on_csv(
    prompt: str,
    *,
    csv_path: str = "ugi_merged_dataset.csv",
    topk_frac: float = 0.01,
    impute_features: Optional[str] = "median",
    aggregate_duplicates: Optional[str] = "mean",
    objective_col: Optional[str] = "yield",
) -> Dict[str, Any]:
    lookup = load_lookup_csv(
        csv_path,
        device=DEVICE,
        dtype=DTYPE,
        impute_features=impute_features,
        aggregate_duplicates=aggregate_duplicates,
        objective_col=objective_col,
    )

    ro_raw = prompt_to_readout(prompt, mins=lookup.mins.detach().cpu().tolist(), maxs=lookup.maxs.detach().cpu().tolist(), d=lookup.d)
    ro_unit = normalize_readout_to_unit_box(ro_raw, lookup.mins, lookup.maxs, feature_names=lookup.feature_names)
    prior = readout_to_prior(ro_unit, feature_names=lookup.feature_names)
    metrics = score_prior_against_truth(prior, lookup.X, lookup.y, topk_frac=topk_frac)

    return {
        "metrics": metrics,
        "readout_raw": ro_raw,
        "readout_unit": ro_unit,
        "n": lookup.n,
        "d": lookup.d,
        "objective": lookup.objective_name,
        "features": lookup.feature_names,
    }


def score_prompt_library_on_csv(
    *,
    csv_path: str = "ugi_merged_dataset.csv",
    topk_frac: float = 0.01,
    objective_col: Optional[str] = "yield",
) -> List[Dict[str, Any]]:
    from bo_readout_prompts import (
        SYS_PROMPTS_BAD,
        SYS_PROMPTS_BEST,
        SYS_PROMPTS_GOOD,
        SYS_PROMPTS_MEDIUM,
        SYS_PROMPTS_PERFECT,
        SYS_PROMPTS_RANDOM,
        SYS_PROMPT_MINIMAL_HUMAN,
    )

    library = {
        "flat": None,
        "best": SYS_PROMPTS_BEST,
        "perfect": SYS_PROMPTS_PERFECT,
        "good": SYS_PROMPTS_GOOD,
        "medium": SYS_PROMPTS_MEDIUM,
        "minimal": SYS_PROMPT_MINIMAL_HUMAN,
        "random": SYS_PROMPTS_RANDOM,
        "bad": SYS_PROMPTS_BAD,
    }

    out: List[Dict[str, Any]] = []
    for name, prompt in library.items():
        if prompt is None:
            lookup = load_lookup_csv(
                csv_path,
                device=DEVICE,
                dtype=DTYPE,
                impute_features="median",
                aggregate_duplicates="mean",
                objective_col=objective_col,
            )
            ro_unit = flat_readout(feature_names=lookup.feature_names)
            prior = readout_to_prior(ro_unit, feature_names=lookup.feature_names)
            metrics = score_prior_against_truth(prior, lookup.X, lookup.y, topk_frac=topk_frac)
            scored = {
                "metrics": metrics,
                "readout_raw": None,
                "readout_unit": ro_unit,
                "n": lookup.n,
                "d": lookup.d,
                "objective": lookup.objective_name,
                "features": lookup.feature_names,
            }
        else:
            scored = score_prompt_on_csv(prompt, csv_path=csv_path, topk_frac=topk_frac, objective_col=objective_col)
        out.append({"name": name, **scored})
    return out


def _print_summary(rows: List[Dict[str, Any]], *, sort_by: str = "pearson_alignment") -> None:
    def key_fn(r: Dict[str, Any]) -> float:
        return float((r.get("metrics") or {}).get(sort_by, float("-inf")))

    rows = sorted(rows, key=key_fn, reverse=True)
    for r in rows:
        m = r["metrics"]
        print(
            f"{r['name']:>8s} | pearson={m['pearson_alignment']:+.3f} "
            f"| spearman={m['spearman_alignment']:+.3f} "
            f"| topk_overlap={m['topk_overlap']:.3f} "
            f"| topk_mean_y={m['topk_mean_y']:.4f}"
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Heuristically score prompt readout alignment vs dataset truth.")
    ap.add_argument("--csv", default="ugi_merged_dataset.csv", help="CSV path (features..., objective last).")
    ap.add_argument("--objective-col", default="yield", help="Objective column name (default: yield).")
    ap.add_argument("--topk-frac", type=float, default=0.01, help="Top-k fraction for overlap/mean metrics.")
    ap.add_argument("--dump-json", action="store_true", help="Print full JSON outputs (readouts + metrics).")
    ap.add_argument("--cuda", action="store_true", help="Use CUDA if available (default: CPU).")
    args = ap.parse_args(list(argv) if argv is not None else None)

    global DEVICE
    if args.cuda and torch.cuda.is_available():
        DEVICE = torch.device("cuda")

    rows = score_prompt_library_on_csv(csv_path=args.csv, topk_frac=args.topk_frac, objective_col=args.objective_col)
    _print_summary(rows)
    if args.dump_json:
        print(json.dumps(rows, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
