"""
cartographer.py
==============

Automated Scientific Discovery (UGI Reaction) — Explorer + Analyst loop.

Two agents:
1) Cartographer (Explorer): Gaussian Process (GP) pure exploration by maximizing predictive uncertainty (variance).
2) Analyst (Synthesizer): LLM periodically summarizes observed data into a qualitative landscape description.

This script is Jupyter-friendly (all core functionality is in functions/classes).
It avoids explicit try/except and explicit raises to keep failures visible as tracebacks.
"""

from __future__ import annotations

# %%
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor

import matplotlib.pyplot as plt

from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls import ExactMarginalLogLikelihood


# --------------------------------------------------------------------
# Optional dependency: score_prompt_via_llm (existing in this repo)
# --------------------------------------------------------------------

if Path(__file__).with_name("prompt_generator.py").exists():
    from prompt_generator import score_prompt_via_llm  # type: ignore
else:
    def score_prompt_via_llm(*, prompt: str, **kwargs: Any) -> Dict[str, Any]:
        return {"metrics": {"ndcg_top_k": 0.8}}


# --------------------------------------------------------------------
# Oracle / Dataset loader (existing in this repo)
# --------------------------------------------------------------------

if Path(__file__).with_name("prompt_generator.py").exists():
    from prompt_generator import load_lookup_csv as load_lookup_csv_oracle  # type: ignore
else:
    # Minimal fallback loader (expects objective column named 'yield')
    def load_lookup_csv_oracle(path: str, *, objective_col: str = "yield"):
        df = pd.read_csv(path)
        feat_cols = [c for c in df.columns if c != objective_col]
        X_raw = torch.tensor(df[feat_cols].to_numpy(dtype=np.float64), dtype=torch.float32)
        y = torch.tensor(df[objective_col].to_numpy(dtype=np.float64), dtype=torch.float32).reshape(-1)
        mins = X_raw.min(dim=0).values
        maxs = X_raw.max(dim=0).values
        rng = (maxs - mins).clamp_min(1e-12)
        X = (X_raw - mins) / rng
        return type(
            "LookupTable",
            (),
            {
                "X_raw": X_raw,
                "y": y,
                "X": X,
                "mins": mins,
                "maxs": maxs,
                "feature_names": feat_cols,
                "objective_name": objective_col,
                "n": int(X.shape[0]),
                "d": int(X.shape[1]),
            },
        )()


# --------------------------------------------------------------------
# Analyst prompt (as requested)
# --------------------------------------------------------------------

ANALYST_SYSTEM_PROMPT_DIRECT_JSON = """You are a Senior Chemist and Data Scientist analyzing experimental data for a UGI reaction.
Your goal is to infer the chemical landscape structure based *only* on the observed data provided below.

The parameters are:
x1: Amine concentration (mM)
x2: Aldehyde concentration (mM)
x3: Isocyanide concentration (mM)
x4: Catalyst amount (equivalents)

You must return a STRICT JSON response defining a Bayesian Prior for Gaussian Process optimization.

Required JSON format:
{
  "effects": {
    "x1": {"effect": "<effect_type>", "scale": <float>, "confidence": <float>, "range_hint": [<low_01>, <high_01>]},
    "x2": {"effect": "<effect_type>", "scale": <float>, "confidence": <float>, "range_hint": [<low_01>, <high_01>]},
    "x3": {"effect": "<effect_type>", "scale": <float>, "confidence": <float>, "range_hint": [<low_01>, <high_01>]},
    "x4": {"effect": "<effect_type>", "scale": <float>, "confidence": <float>, "range_hint": [<low_01>, <high_01>]}
  },
  "interactions": [
    {"vars": ["xi", "xj"], "type": "<synergy|tradeoff>", "scale": <float>, "confidence": <float>, "note": "<brief explanation>"}
  ],
  "bumps": [
    {"mu": [<x1_01>, <x2_01>, <x3_01>, <x4_01>], "sigma": [<s1>, <s2>, <s3>, <s4>], "amp": <float>}
  ]
}

Field specifications:
• effect: MUST be one of {"increasing", "decreasing", "nonmonotone-peak", "nonmonotone-valley", "flat"}
  - increasing: higher variable values → higher yield
  - decreasing: higher variable values → lower yield
  - nonmonotone-peak: yield peaks at intermediate value
  - nonmonotone-valley: yield dips at intermediate value
  - flat: no clear relationship
• scale: effect magnitude, range [0.0, 1.0] (0=negligible, 1=very strong)
• confidence: certainty of assessment, range [0.0, 1.0] (0=very uncertain, 1=highly confident)
• range_hint: [low, high] in NORMALIZED [0,1] space (not raw units!)
  - Convert raw ranges to [0,1] using: (value - min_obs) / (max_obs - min_obs)
• interactions: list 1-3 most important pairwise effects (can be empty [])
  - type: "synergy" (both high → good) or "tradeoff" (opposite directions → good)
  - scale/confidence: same interpretation as effects
• bumps: list 0-2 Gaussian hotspots in normalized [0,1]^4 space
  - mu: center coordinates [x1, x2, x3, x4] in [0,1] space
  - sigma: width per dimension (typically 0.1-0.3 for moderate uncertainty)
  - amp: amplitude (typically 0.05-0.15)

CRITICAL RULES:
1. Base ALL conclusions ONLY on observed data - ignore external chemical knowledge
2. Use LOW confidence (<0.5) when n_samples < 50 or patterns are weak
3. Use scale=0.0, confidence=0.0 (or effect="flat") when no pattern is evident
4. For range_hint, analyze where the effect applies in the observed data range
5. Only include interactions with clear evidence (e.g., high yield requires BOTH variables high)
6. Only place bumps near observed high-yield clusters
7. Be conservative: underestimating patterns is safer than hallucinating them
8. With sparse data (<30 samples), prefer simple models: fewer interactions, broader bumps

Return ONLY valid JSON with no additional commentary."""


def make_openai_client(*, api_key: Optional[str] = None):
    """
    Creates an OpenAI client with httpx verify=False (as requested).
    Network calls may require approvals depending on your environment.
    """
    import httpx  # type: ignore
    from openai import OpenAI  # type: ignore

    return OpenAI(api_key=api_key, http_client=httpx.Client(verify=False))


def _format_observations_for_analyst_enhanced(
    X_raw_obs: np.ndarray,
    y_obs: np.ndarray,
    feature_names: List[str],
    *,
    max_rows: int = 30,
) -> str:
    """Enhanced data formatting with binned analysis and better statistics."""
    df = pd.DataFrame(X_raw_obs, columns=feature_names)
    df["yield"] = y_obs
    df = df.reset_index(drop=True)

    n = int(len(df))
    parts: List[str] = []
    parts.append(f"=== OBSERVED DATA SUMMARY (n={n} samples) ===\n")
    parts.append(f"Yield statistics:")
    parts.append(f"  min={df['yield'].min():.6f}, median={df['yield'].median():.6f}, mean={df['yield'].mean():.6f}, max={df['yield'].max():.6f}")
    parts.append(f"  std={df['yield'].std():.6f}, Q1={df['yield'].quantile(0.25):.6f}, Q3={df['yield'].quantile(0.75):.6f}\n")

    # Observed ranges and normalized space reminder
    parts.append("Variable ranges in RAW units (for your reference):")
    for c in feature_names:
        parts.append(f"  {c}: [{df[c].min():.4f}, {df[c].max():.4f}]")
    parts.append("\nREMINDER: Output range_hint and bump mu/sigma in NORMALIZED [0,1] space!")
    parts.append("  Normalize: (value - min) / (max - min)\n")

    # Correlations (small-sample warning)
    if n >= 3:
        corr = df[feature_names + ["yield"]].corr(numeric_only=True)["yield"].drop("yield")
        parts.append("Pearson correlations with yield:")
        for name in feature_names:
            c_val = float(corr.get(name, np.nan))
            parts.append(f"  {name}: {c_val:+.4f}")
        if n < 30:
            parts.append("  WARNING: Small sample size - correlations may be unstable!\n")
        else:
            parts.append("")

    # Binned analysis for each variable (if enough data)
    if n >= 15:
        parts.append("=== BINNED ANALYSIS (split at median) ===")
        for var in feature_names:
            med = df[var].median()
            low_mask = df[var] <= med
            high_mask = df[var] > med
            if low_mask.sum() > 0 and high_mask.sum() > 0:
                y_low_mean = df.loc[low_mask, "yield"].mean()
                y_high_mean = df.loc[high_mask, "yield"].mean()
                diff = y_high_mean - y_low_mean
                parts.append(f"{var}: low half mean_yield={y_low_mean:.4f}, high half mean_yield={y_high_mean:.4f}, diff={diff:+.4f}")
        parts.append("")

    # Top and bottom samples
    k = int(min(max_rows // 2, max(5, n // 3)))
    if k > 0:
        parts.append(f"=== TOP {k} YIELD SAMPLES ===")
        top = df.sort_values("yield", ascending=False).head(k)
        parts.append(top.to_csv(index=False))

        parts.append(f"=== BOTTOM {k} YIELD SAMPLES ===")
        bot = df.sort_values("yield", ascending=True).head(k)
        parts.append(bot.to_csv(index=False))

    return "\n".join(parts)


def synthesize_landscape_readout_direct(
    X_raw_obs: np.ndarray,
    y_obs: np.ndarray,
    feature_names: List[str],
    *,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> Dict[str, Any]:
    """
    Direct JSON generation: Analyst produces readout JSON in one step (no intermediate prose).
    This eliminates the lossy two-step translation process.
    """
    import json
    client = make_openai_client(api_key=api_key)
    payload = _format_observations_for_analyst_enhanced(X_raw_obs, y_obs, feature_names, max_rows=30)
    messages = [
        {"role": "system", "content": ANALYST_SYSTEM_PROMPT_DIRECT_JSON},
        {"role": "user", "content": f"Analyze these observed samples and return the JSON prior:\n\n{payload}"},
    ]

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=messages,
        )
        raw_text = str(resp.choices[0].message.content or "").strip()
        readout = json.loads(raw_text)

        # Minimal validation
        if "effects" not in readout:
            readout["effects"] = {}
        if "interactions" not in readout:
            readout["interactions"] = []
        if "bumps" not in readout:
            readout["bumps"] = []

        return readout
    except Exception as e:
        # Fallback: return flat prior on error
        print(f"WARNING: Direct JSON generation failed: {e}")
        return {
            "effects": {name: {"effect": "flat", "scale": 0.0, "confidence": 0.0} for name in feature_names},
            "interactions": [],
            "bumps": [],
        }


def synthesize_landscape_description(
    X_raw_obs: np.ndarray,
    y_obs: np.ndarray,
    feature_names: List[str],
    *,
    api_key: Optional[str] = None,
    model: str = "gpt-4o",
    temperature: float = 0.2,
    max_tokens: int = 1200,
) -> str:
    """
    LEGACY: Analyst generates qualitative description (old two-step approach).
    Increased max_tokens to prevent truncation.
    Consider using synthesize_landscape_readout_direct() instead.
    """
    client = make_openai_client(api_key=api_key)
    payload = _format_observations_for_analyst_enhanced(X_raw_obs, y_obs, feature_names, max_rows=30)
    messages = [
        {"role": "system", "content": ANALYST_SYSTEM_PROMPT_DIRECT_JSON[:500]},  # Use truncated version for backward compat
        {"role": "user", "content": "Analyze these observed samples:\n\n" + payload},
    ]
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=messages,
    )
    return str(resp.choices[0].message.content or "").strip()


def build_scoring_prompt_from_description(description: str) -> str:
    """
    Wrap the Analyst's natural-language description into the JSON readout prompt
    expected by prompt_generator.score_prompt_via_llm.
    """
    from bo_readout_prompts import PROMPT_HEADER, UGI_DATASET_CONTEXT, UGI_RESPONSE_FORMAT

    body = f"""
You are given an analyst's qualitative landscape description intended to serve as a Bayesian Prior:

\"\"\"
{description.strip()}
\"\"\"

Translate that description into the STRICT JSON response format below.
If the description is uncertain about a trend, encode that uncertainty via lower confidence/scale.
"""
    return f"{PROMPT_HEADER}\n{UGI_DATASET_CONTEXT}\n{body.strip()}\n\n{UGI_RESPONSE_FORMAT}\n"


def extract_ndcg_score(score_payload: Dict[str, Any]) -> float:
    metrics = score_payload.get("metrics") if isinstance(score_payload, dict) else None
    if not isinstance(metrics, dict):
        return float("nan")
    for key in ["ndcg_top_k", "ndcg_at_topk", "ndcg", "ndcg@k"]:
        if key in metrics:
            return float(metrics[key])
    return float("nan")


@dataclass
class Cartographer:
    """
    Pure exploration agent on a finite lookup table:
    selects the next unseen candidate with maximum GP posterior variance.
    """

    lookup: Any
    seed: int = 0
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.double

    seen: set[int] = field(default_factory=set)
    X_obs: Tensor = field(default_factory=lambda: torch.empty((0, 1)))
    Y_obs: Tensor = field(default_factory=lambda: torch.empty((0,)))
    history: List[Dict[str, Any]] = field(default_factory=list)

    def initialize(self, *, n_init: int = 6) -> None:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(self.seed))
        n = int(self.lookup.n)

        init = torch.randperm(n, generator=g)[: int(min(n_init, n))].tolist()
        for i, idx in enumerate(init):
            self.observe(int(idx), iteration=i - len(init), tag="init")

    def observe(self, idx: int, *, iteration: int, tag: str = "obs") -> None:
        idx = int(idx)
        if idx in self.seen:
            return
        self.seen.add(idx)

        x_unit = self.lookup.X[idx].to(device=self.device, dtype=self.dtype).unsqueeze(0)
        y = float(self.lookup.y[idx].item())
        y_t = torch.tensor([y], device=self.device, dtype=self.dtype)

        if self.X_obs.numel() == 0 or self.X_obs.ndim != 2:
            self.X_obs = x_unit.clone()
        else:
            self.X_obs = torch.cat([self.X_obs, x_unit], dim=0)
        if self.Y_obs.numel() == 0:
            self.Y_obs = y_t.clone()
        else:
            self.Y_obs = torch.cat([self.Y_obs, y_t], dim=0)

        raw = self.lookup.X_raw[idx].detach().cpu().numpy().tolist()
        rec = {
            "iter": int(iteration),
            "idx": int(idx),
            "tag": str(tag),
            "y": float(y),
            "best_so_far": float(self.Y_obs.max().item()) if self.Y_obs.numel() else float("nan"),
        }
        for j, val in enumerate(raw):
            rec[f"x{j+1}"] = float(val)
        self.history.append(rec)

    def fit_gp(self) -> SingleTaskGP:
        X = self.X_obs.to(device=self.device, dtype=self.dtype)
        Y = self.Y_obs.to(device=self.device, dtype=self.dtype).unsqueeze(-1)
        gp = SingleTaskGP(X, Y, outcome_transform=Standardize(m=1)).to(self.device)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp).to(self.device)
        fit_gpytorch_mll(mll)
        return gp

    def suggest(self, gp: SingleTaskGP) -> Tuple[int, float]:
        rem = [i for i in range(int(self.lookup.n)) if i not in self.seen]
        if len(rem) == 0:
            return -1, float("nan")
        X_pool = self.lookup.X[rem].to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            post = gp.posterior(X_pool.unsqueeze(1))
            var = post.variance.reshape(-1).clamp_min(1e-12)
            j = int(torch.argmax(var).item())
            return int(rem[j]), float(torch.sqrt(var[j]).item())

    def step(self, *, iteration: int) -> Dict[str, Any]:
        gp = self.fit_gp()
        idx, std = self.suggest(gp)
        if idx < 0:
            return {"iter": int(iteration), "idx": -1, "status": "done"}
        self.observe(idx, iteration=iteration, tag="explore")
        self.history[-1]["pred_std"] = float(std)
        return dict(self.history[-1])

    def run(self, *, budget: int = 50, n_init: int = 6) -> pd.DataFrame:
        if len(self.seen) == 0:
            self.initialize(n_init=n_init)
        for t in range(int(budget)):
            self.step(iteration=t)
        return pd.DataFrame(self.history)


def run_cartographer_analyst_pipeline_improved(
    *,
    csv_path: str = "ugi_merged_dataset.csv",
    objective_col: str = "yield",
    budget: int = 50,
    n_init: int = 6,
    seed: int = 0,
    analyst_every: int = 10,
    analyst_model: str = "gpt-4o",
    api_key: Optional[str] = None,
    use_direct_json: bool = True,) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    IMPROVED pipeline with direct JSON generation (no two-step translation).

    - Cartographer explores for `budget` iterations
    - Every `analyst_every` iterations: Analyst generates JSON readout directly
    - Score the readout against ground truth using NDCG

    Returns:
      (history_df, knowledge_df)

    Changes from original:
      1. Direct JSON generation eliminates prose→JSON translation loss
      2. No ground truth contamination in prompts
      3. Enhanced data representation with binned analysis
      4. Increased max_tokens to prevent truncation
    """
    from readout_schema import readout_to_prior
    from prior_gp import alignment_on_obs
    from prompt_generator import load_lookup_csv, score_prior_against_truth, normalize_readout_to_unit_box

    lookup = load_lookup_csv_oracle(csv_path, objective_col=objective_col)
    cart = Cartographer(lookup=lookup, seed=seed, device=torch.device("cpu"), dtype=torch.double)
    cart.initialize(n_init=n_init)

    # Load full dataset for scoring
    lookup_full = load_lookup_csv(csv_path, objective_col=objective_col, device=torch.device("cpu"), dtype=torch.float32)

    knowledge: List[Dict[str, Any]] = []

    for t in range(int(budget)):
        cart.step(iteration=t)

        if analyst_every > 0 and ((t + 1) % int(analyst_every) == 0):
            X_raw_obs = lookup.X_raw[list(cart.seen)].detach().cpu().numpy()
            y_obs = lookup.y[list(cart.seen)].detach().cpu().numpy().reshape(-1)
            analyst_feature_names = ["x1", "x2", "x3", "x4"] if int(getattr(lookup, "d", 0)) == 4 else list(getattr(lookup, "feature_names", []))

            if use_direct_json:
                # NEW: Direct JSON generation (one-step)
                readout_raw = synthesize_landscape_readout_direct(
                    X_raw_obs,
                    y_obs,
                    feature_names=analyst_feature_names,
                    api_key=api_key,
                    model=analyst_model,
                    temperature=0.2,
                    max_tokens=1500,
                )
                # Normalize to unit box
                readout_unit = normalize_readout_to_unit_box(
                    readout_raw,
                    lookup_full.mins,
                    lookup_full.maxs,
                    feature_names=analyst_feature_names
                )
                # Convert to Prior and score
                prior = readout_to_prior(readout_unit, feature_names=analyst_feature_names)
                metrics = score_prior_against_truth(
                    prior,
                    lookup_full.X,
                    lookup_full.y,
                    topk_frac=0.01,
                    tail_q=0.8
                )
                ndcg = metrics.get("ndcg_at_topk", float("nan"))
                description = f"Direct JSON readout (effects: {len(readout_unit.get('effects', {}))}, interactions: {len(readout_unit.get('interactions', []))}, bumps: {len(readout_unit.get('bumps', []))})"

                knowledge.append({
                    "iter": int(t),
                    "n_obs": int(len(cart.seen)),
                    "ndcg": float(ndcg),
                    "description": description,
                    "readout": readout_unit,
                    "metrics": metrics,
                })
            else:
                # LEGACY: Two-step approach (kept for comparison)
                description = synthesize_landscape_description(
                    X_raw_obs,
                    y_obs,
                    feature_names=analyst_feature_names,
                    api_key=api_key,
                    model=analyst_model,
                )
                scoring_prompt = build_scoring_prompt_from_description(description)
                score_payload = score_prompt_via_llm(
                    prompt=scoring_prompt,
                    csv_path=csv_path,
                    objective_col=objective_col,
                    model=analyst_model,
                    api_key=api_key,
                )
                ndcg = extract_ndcg_score(score_payload)
                knowledge.append({
                    "iter": int(t),
                    "n_obs": int(len(cart.seen)),
                    "ndcg": float(ndcg),
                    "description": description,
                })

    hist_df = pd.DataFrame(cart.history)
    knowledge_df = pd.DataFrame(knowledge)
    return hist_df, knowledge_df


def run_cartographer_analyst_pipeline(
    *,
    csv_path: str = "ugi_merged_dataset.csv",
    objective_col: str = "yield",
    budget: int = 50,
    n_init: int = 6,
    seed: int = 0,
    analyst_every: int = 10,
    analyst_model: str = "gpt-4o",
    scorer_model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    LEGACY pipeline (kept for backward compatibility).
    Consider using run_cartographer_analyst_pipeline_improved() instead.

    Main loop:
    - Cartographer explores for `budget` iterations.
    - Every `analyst_every` iterations: Analyst summarizes and we score knowledge via score_prompt_via_llm (NDCG).
    Returns:
      (history_df, knowledge_df)
    """
    lookup = load_lookup_csv_oracle(csv_path, objective_col=objective_col)
    cart = Cartographer(lookup=lookup, seed=seed, device=torch.device("cpu"), dtype=torch.double)
    cart.initialize(n_init=n_init)

    knowledge: List[Dict[str, Any]] = []

    for t in range(int(budget)):
        cart.step(iteration=t)

        if analyst_every > 0 and ((t + 1) % int(analyst_every) == 0):
            X_raw_obs = lookup.X_raw[list(cart.seen)].detach().cpu().numpy()
            y_obs = lookup.y[list(cart.seen)].detach().cpu().numpy().reshape(-1)
            analyst_feature_names = ["x1", "x2", "x3", "x4"] if int(getattr(lookup, "d", 0)) == 4 else list(getattr(lookup, "feature_names", []))
            description = synthesize_landscape_description(
                X_raw_obs,
                y_obs,
                feature_names=analyst_feature_names,
                api_key=api_key,
                model=analyst_model,
            )
            scoring_prompt = build_scoring_prompt_from_description(description)
            score_payload = score_prompt_via_llm(
                prompt=scoring_prompt,
                csv_path=csv_path,
                objective_col=objective_col,
                model=scorer_model,
                api_key=api_key,
            )
            ndcg = extract_ndcg_score(score_payload)
            knowledge.append(
                {
                    "iter": int(t),
                    "n_obs": int(len(cart.seen)),
                    "ndcg": float(ndcg),
                    "description": description,
                }
            )

    hist_df = pd.DataFrame(cart.history)
    knowledge_df = pd.DataFrame(knowledge)
    return hist_df, knowledge_df


def plot_knowledge_curve(knowledge_df: pd.DataFrame, *, ax: Optional[plt.Axes] = None, title: str = "Knowledge Score (NDCG)") -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 4.0))
    if knowledge_df is None or knowledge_df.empty:
        ax.set_title(title + " (no checkpoints)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("NDCG")
        ax.grid(True, alpha=0.35)
        return ax
    ax.plot(knowledge_df["iter"], knowledge_df["ndcg"], marker="o")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("NDCG")
    ax.grid(True, alpha=0.35)
    plt.tight_layout()
    return ax


# --------------------------------------------------------------------
# Usage (Jupyter / Script)
# --------------------------------------------------------------------
#
# Jupyter:
#   from cartographer import run_cartographer_analyst_pipeline, plot_knowledge_curve
#   hist_df, knowledge_df = run_cartographer_analyst_pipeline(
#       csv_path="ugi_merged_dataset.csv",
#       budget=50,
#       n_init=6,
#       seed=0,
#       analyst_every=10,
#       analyst_model="gpt-4o",
#       scorer_model="gpt-4o-mini",
#       api_key=None,  # or set your API key
#   )
#   plot_knowledge_curve(knowledge_df)
#
# Script:
#   python cartographer.py
#
if __name__ == "__main__":
    # hist_df, knowledge_df = run_cartographer_analyst_pipeline_improved(
    #     csv_path="ugi_merged_dataset.csv",
    #     budget=100,
    #     n_init=10,
    #     seed=1,
    #     analyst_every=20,
    #     analyst_model="gpt-4o",
    #     scorer_model="gpt-4o-mini",
    #     api_key=None,
    # )
    # print(knowledge_df[["iter", "n_obs", "ndcg"]])
    # plot_knowledge_curve(knowledge_df)
    # plt.show()



    hist, knowledge = run_cartographer_analyst_pipeline_improved(                                                                                              
        csv_path="ugi_merged_dataset.csv",                                                                                                                     
        budget=200,                                                                                                                                            
        analyst_every=20,  # Check every 20 iterations (was 10)                                                                                                
        use_direct_json=True,  # NEW: Recommended                                                                                                              
        api_key=None,  # Or your OpenAI key                                                                                                                    
    )    

#%%