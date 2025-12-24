"""
Data distillation pipeline for deriving a prior from historical CSV data.

Notebook-friendly: no CLI entrypoints, just importable helpers.
"""
#%%
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import json

import numpy as np
import pandas as pd

MIN_SAMPLE_ROWS = 5


def sample_legacy_data(
    csv_path: str,
    *,
    fraction: float = 0.10,
    seed: int = 0,
) -> pd.DataFrame:
    """Load the CSV and sample a fraction of rows as historical data."""
    if fraction <= 0.0 or fraction > 1.0:
        raise ValueError("fraction must be in (0, 1].")
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV is empty; no rows to sample.")
    n_total = int(df.shape[0])
    sample_n = int(np.ceil(float(fraction) * n_total))
    sample_n = max(sample_n, MIN_SAMPLE_ROWS)
    sample_n = min(sample_n, n_total)
    return df.sample(n=sample_n, random_state=int(seed))


def _resolve_feature_names(
    df: pd.DataFrame,
    feature_names: Optional[Iterable[str]],
    target_col: str,
) -> List[str]:
    if feature_names is not None:
        return [str(c) for c in feature_names]
    cols = df.select_dtypes(include="number").columns.tolist()
    if target_col in cols:
        cols.remove(target_col)
    if not cols:
        raise ValueError("No numeric feature columns found to summarize.")
    return cols


def format_data_for_llm(
    df_sampled: pd.DataFrame,
    *,
    feature_names: Optional[Iterable[str]],
    target_col: str = "yield",
    top_k: int = 20,
    bottom_k: int = 20,
    include_summary_stats: bool = False,
) -> str:
    """Format only top/bottom cases from a sampled portion into a prompt string."""
    top_df, bottom_df, feat_names, df_clean = select_top_bottom_cases(
        df_sampled,
        feature_names=feature_names,
        target_col=target_col,
        top_k=top_k,
        bottom_k=bottom_k,
    )
    required_cols = [target_col, *feat_names]
    n_rows = int(df_clean.shape[0])

    def _fmt_row(row: pd.Series) -> str:
        row_id = row.name
        parts = [f"{target_col}={float(row[target_col]):.4g}"]
        for name in feat_names:
            parts.append(f"{name}={float(row[name]):.4g}")
        return f"row {row_id}: " + ", ".join(parts)

    lines: List[str] = []
    lines.append(f"Sampled portion size: {n_rows} historical experiments.")
    lines.append(f"Features: {', '.join(feat_names)}")
    lines.append(f"Target: {target_col}")
    if include_summary_stats:
        stats = df_clean[required_cols].agg(["mean", "min", "max"]).T
        lines.append("")
        lines.append("Summary statistics (mean/min/max):")
        for col in required_cols:
            mean = float(stats.loc[col, "mean"])
            vmin = float(stats.loc[col, "min"])
            vmax = float(stats.loc[col, "max"])
            lines.append(f"- {col}: mean={mean:.4g}, min={vmin:.4g}, max={vmax:.4g}")

    lines.append("")
    lines.append(f"Best yields were found at (top {len(top_df)}):")
    for _, row in top_df.iterrows():
        lines.append(f"- {_fmt_row(row)}")

    lines.append("")
    lines.append(f"Worst yields (Failures) were found at (bottom {len(bottom_df)}):")
    if bottom_df.empty:
        lines.append("- None (sample too small for distinct failures).")
    else:
        for _, row in bottom_df.iterrows():
            lines.append(f"- {_fmt_row(row)}")

    return "\n".join(lines)


def select_top_bottom_cases(
    df_sampled: pd.DataFrame,
    *,
    feature_names: Optional[Iterable[str]],
    target_col: str = "yield",
    top_k: int = 20,
    bottom_k: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame]:
    """Return top/bottom rows from a sampled portion for LLM prompting."""
    if target_col not in df_sampled.columns:
        raise ValueError(f"target_col={target_col!r} not found in sampled data.")

    feat_names = _resolve_feature_names(df_sampled, feature_names, target_col)
    required_cols = [target_col, *feat_names]
    missing = [c for c in required_cols if c not in df_sampled.columns]
    if missing:
        raise ValueError(f"Missing columns in sampled data: {missing}")

    df_clean = df_sampled.dropna(subset=required_cols)
    if df_clean.empty:
        raise ValueError("Sampled data has no valid rows after dropping missing values.")

    n_rows = int(df_clean.shape[0])
    top_n = min(int(top_k), n_rows)
    bottom_n = min(int(bottom_k), n_rows)

    df_sorted = df_clean.sort_values(target_col, ascending=False)
    top_df = df_sorted.head(top_n)
    bottom_df = df_sorted.sort_values(target_col, ascending=True).head(bottom_n)
    bottom_df = bottom_df.loc[~bottom_df.index.isin(top_df.index)]

    return top_df, bottom_df, feat_names, df_clean


def _coerce_constraints_schema(
    readout: Dict[str, Any],
    *,
    default_penalty: float = 5.0,
) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(readout or {})
    constraints_in = out.get("constraints") or []
    constraints_out = []
    for c in constraints_in:
        if not isinstance(c, dict):
            continue
        var = c.get("var")
        r = c.get("range")
        if var is None or not isinstance(r, (list, tuple)) or len(r) != 2:
            continue
        try:
            lo = float(r[0])
            hi = float(r[1])
        except (TypeError, ValueError):
            continue
        if hi < lo:
            lo, hi = hi, lo
        penalty_raw = c.get("penalty", c.get("weight", default_penalty))
        try:
            penalty = float(penalty_raw)
        except (TypeError, ValueError):
            penalty = float(default_penalty)
        c_out = dict(c)
        c_out["var"] = str(var)
        c_out["range"] = [lo, hi]
        c_out["penalty"] = penalty
        if not c_out.get("reason"):
            c_out["reason"] = "low-yield region"
        constraints_out.append(c_out)
    out["constraints"] = constraints_out
    return out


def _extract_features_from_summary(formatted_text: str) -> List[str]:
    for line in formatted_text.splitlines():
        if line.lower().startswith("features:"):
            raw = line.split(":", 1)[1].strip()
            if not raw:
                return []
            return [token.strip() for token in raw.split(",") if token.strip()]
    return []


def _ensure_effects_for_features(
    readout: Dict[str, Any],
    *,
    feature_names: Iterable[str],
) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(readout or {})
    effects = out.get("effects") or {}
    if not isinstance(effects, dict):
        effects = {}
    for name in feature_names:
        if name not in effects:
            effects[name] = {"effect": "flat", "scale": 0.0, "confidence": 0.0}
    out["effects"] = effects
    return out


def _range_span(range_pair: Iterable[float]) -> float:
    vals = list(range_pair)
    if len(vals) != 2:
        return 0.0
    return float(vals[1] - vals[0])


def _compute_ranges(
    df: pd.DataFrame,
    *,
    feature_names: List[str],
    target_col: str,
    top_k: int,
) -> tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    required = [target_col, *feature_names]
    df_clean = df.dropna(subset=required)
    if df_clean.empty:
        return {}, {}

    full_ranges: Dict[str, List[float]] = {}
    for name in feature_names:
        series = pd.to_numeric(df_clean[name], errors="coerce").dropna()
        if series.empty:
            continue
        full_ranges[name] = [float(series.min()), float(series.max())]

    top_k = max(1, min(int(top_k), int(df_clean.shape[0])))
    top_df = df_clean.sort_values(target_col, ascending=False).head(top_k)

    top_ranges: Dict[str, List[float]] = {}
    for name in feature_names:
        series = pd.to_numeric(top_df[name], errors="coerce").dropna()
        if series.empty:
            continue
        top_ranges[name] = [float(series.min()), float(series.max())]
    return top_ranges, full_ranges


def _tighten_range_hints(
    readout: Dict[str, Any],
    *,
    df_sampled: pd.DataFrame,
    feature_names: List[str],
    target_col: str,
    top_k: int = 10,
    max_span_frac: float = 0.9,
    fill_missing: bool = True,
) -> Dict[str, Any]:
    top_ranges, full_ranges = _compute_ranges(
        df_sampled, feature_names=feature_names, target_col=target_col, top_k=top_k
    )
    if not top_ranges or not full_ranges:
        return readout

    out: Dict[str, Any] = dict(readout or {})
    effects = out.get("effects") or {}
    if not isinstance(effects, dict):
        return out

    for name in feature_names:
        spec = effects.get(name)
        if not isinstance(spec, dict):
            continue

        top_range = top_ranges.get(name)
        full_range = full_ranges.get(name)
        if not top_range or not full_range:
            continue

        full_span = _range_span(full_range)
        if full_span <= 0:
            continue

        range_hint = spec.get("range_hint")
        use_top = False

        if range_hint is None and fill_missing:
            use_top = True
        elif isinstance(range_hint, (list, tuple)) and len(range_hint) == 2:
            try:
                lo = float(range_hint[0])
                hi = float(range_hint[1])
            except (TypeError, ValueError):
                use_top = True
            else:
                if hi < lo:
                    lo, hi = hi, lo
                span_frac = (hi - lo) / full_span
                if span_frac >= float(max_span_frac):
                    use_top = True
        else:
            use_top = True

        if use_top:
            lo, hi = float(top_range[0]), float(top_range[1])
            if hi <= lo:
                eps = max(0.01 * full_span, 1e-6)
                center = lo
                lo = max(full_range[0], center - eps)
                hi = min(full_range[1], center + eps)
            spec["range_hint"] = [lo, hi]

    out["effects"] = effects
    return out


def extract_prior_from_statistics(
    formatted_text: str,
    *,
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 800,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Call the LLM to extract a readout JSON from the formatted summary."""
    try:
        import httpx  # type: ignore
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise RuntimeError("LLM mode requires `openai` and `httpx` to be installed.") from exc

    system_prompt = f"""
You are an Expert Chemometrician and Bayesian Optimization Architect.
Your goal is to extract a "Search Prior" from sparse pilot data to accelerate optimization.
You must balance "Exploitation" (finding peaks) with "Safety" (avoiding dead zones).

DATA CONTEXT:
{formatted_text}

--- ANALYSIS PROTOCOL ---

1. IDENTIFY CONSTRAINTS (Negative Knowledge):
   - Look at the "Failure Cases" (Low Yields).
   - Is there a specific variable range that *consistently* appears in failures but NEVER in successes?
   - If yes, define a "constraint".
   - Set 'penalty' high (e.g., 8.0-10.0) for regions that look chemically invalid or dead.
   - Set 'penalty' moderate (e.g., 2.0-5.0) for regions that are just suboptimal.

2. IDENTIFY BUMPS (Positive Knowledge):
   - Look at the single absolute Best Result in the summary.
   - Create a "bump" centered exactly at those coordinates (`mu`).
   - Use `sigma` to define how wide this peak might be (use ~10% of the variable range if unsure).
   - Set `amp` based on the yield (e.g., if Yield=0.9, amp=0.2; if Yield=0.2, amp=0.05).

3. IDENTIFY TRENDS (Global Effects):
   - Compare "Success Cases" vs. "Failure Cases" generally.
   - If a variable is consistently high in successes and low in failures -> "increasing".
   - If a variable is consistently low in successes -> "decreasing".
   - If successes happen in the middle range -> "nonmonotone-peak".
   - If no clear pattern appears, omit it or set effect to "flat".
   - Set 'confidence' based on consistency: 0.9 if the trend has no exceptions, 0.4 if noisy.

--- JSON OUTPUT SCHEMA ---

Return STRICT JSON. No prose.
{{
  "effects": {{
    "<feature_name>": {{
      "effect": "increasing|decreasing|nonmonotone-peak|flat",
      "scale": <float 0.1 to 2.0, strength of trend slope>,
      "confidence": <float 0.0 to 1.0, reliability of data>,
      "range_hint": [<low_float>, <high_float>] (Optional: focus region)
    }}
  }},
  "bumps": [
    {{
      "mu": [<val_x1>, <val_x2>...],
      "sigma": [<width_x1>, <width_x2>...] (in raw units),
      "amp": <float 0.0 to 0.5>
    }}
  ],
  "constraints": [
    {{
      "var": "<feature_name>",
      "range": [<low_float>, <high_float>],
      "penalty": <float, typically 5.0 to 10.0>,
      "reason": "<short string explaining why>"
    }}
  ]
}}

--- CRITICAL RULES ---
1. RAW UNITS ONLY: Do not normalize. If input is '150 degC', use 150.0.
2. PRECISE BUMPS: The 'bumps' list MUST contain at least one entry centered on the best observed row.
3. CONSERVATIVE CONSTRAINTS: Only constrain a region if the evidence of failure is strong.
4. Feature names must match exactly as provided in the summary
""".strip()

    client = OpenAI(api_key=api_key, http_client=httpx.Client(verify=False))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Return STRICT JSON only (no prose)."},
    ]

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
        raw_text = getattr(resp, "output_text", None) or str(resp)

    if not raw_text:
        raise RuntimeError("OpenAI returned empty response.")

    try:
        readout = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model did not return valid JSON. Raw text:\n{raw_text}") from exc

    feature_names = _extract_features_from_summary(formatted_text)
    readout = _ensure_effects_for_features(readout, feature_names=feature_names)
    return _coerce_constraints_schema(readout)


def get_data_derived_prior(
    *,
    csv_path: str,
    fraction: float = 0.10,
    seed: int = 2,
    feature_names: Optional[Iterable[str]] = None,
    target_col: str = "yield",
    model: str = "gpt-4o-mini",
    temperature: float = 0.2,
    max_tokens: int = 800,
    api_key: Optional[str] = None,
    top_k: int = 20,
    bottom_k: int = 20,
    include_summary_stats: bool = False,
    tighten_range_hint: bool = True,
    range_hint_top_k: Optional[int] = None,
    range_hint_max_span_frac: float = 0.9,
    fill_missing_range_hint: bool = True,
) -> Dict[str, Any]:
    """Run sampling + formatting + LLM extraction, returning a readout dict."""
    sampled = sample_legacy_data(csv_path, fraction=fraction, seed=seed)
    feat_names = _resolve_feature_names(sampled, feature_names, target_col)
    formatted = format_data_for_llm(
        sampled,
        feature_names=feat_names,
        target_col=target_col,
        top_k=top_k,
        bottom_k=bottom_k,
        include_summary_stats=include_summary_stats,
    )
    readout = extract_prior_from_statistics(
        formatted,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )
    if tighten_range_hint:
        top_k_for_ranges = range_hint_top_k if range_hint_top_k is not None else top_k
        readout = _tighten_range_hints(
            readout,
            df_sampled=sampled,
            feature_names=feat_names,
            target_col=target_col,
            top_k=top_k_for_ranges,
            max_span_frac=range_hint_max_span_frac,
            fill_missing=fill_missing_range_hint,
        )
    return readout
#%%

get_data_derived_prior(fraction=0.9, csv_path="ugi_merged_dataset.csv")
# %%
