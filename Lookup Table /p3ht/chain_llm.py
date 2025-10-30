"""
chain_llm.py — 3-stage LLM chain for dynamic prior shaping

Pipeline
--------
Raw Context (from GP + history) → [LLM-1: Analyzer] Standardized Summary
                              → [LLM-2: Pattern]  Hypotheses
                              → [LLM-3: Generator] Final Readout (effects/interactions/bumps + m0_weight)

All three functions return strict JSON dicts. If parsing fails, they return safe fallbacks.
You can swap prompts/models later; defaults are chosen to balance cost/quality.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional

import json

# --------- Default system prompts (concise but instructive) ---------

SYS_PROMPT_ANALYZER_V1 = (
    "You are LLM-1 (Analyzer). Your job is to convert raw Bayesian Optimization telemetry into a\n"
    "STRICT JSON summary called 'status_summary_v1'.\n\n"
    "Input you will receive (as JSON):\n"
    "- 'context': numeric features from a GP surrogate over [0,1]^2: top_k EI points, top_k variance points,\n"
    "  incumbent x*, density heatmap (coarse grid visitation counts).\n"
    "- 'recent': a short list of the last evaluations (iter, x1, x2, y, best_so_far).\n\n"
    "Definitions:\n"
    "- EI hotspots: locations with large expected improvement.\n"
    "- Variance hotspots: high posterior uncertainty.\n"
    "- Density heatmap: where we've already sampled a lot.\n"
    "- Stagnation: no improvement in the last few iterations.\n\n"
    "Output schema (STRICT JSON, no prose):\n"
    "{\n"
    "  'iter': <int or null>,\n"
    "  'best_so_far': <float>,\n"
    "  'imp_last_k': <float or null>,\n"
    "  'stagnation': <bool>,\n"
    "  'incumbent': {'x':[x1,x2],'y':<float>},\n"
    "  'top_ei': [{'x':[x1,x2],'score':<float>}, ... up to 5],\n"
    "  'top_var': [{'x':[x1,x2],'score':<float>}, ... up to 5],\n"
    "  'dense_cells': [{'ij':[i,j],'n':<int>}, ... up to 6],\n"
    "  'notes': [short bullet strings, max 5]\n"
    "}\n\n"
    "Rules:\n"
    "- Keep numeric values in [0,1] for coordinates.\n"
    "- 'notes' should be short analytic hints (e.g., 'EI ridge near NE boundary').\n"
    "- No extra keys. No explanations outside JSON."
)


SYS_PROMPT_HYPOTHESES_V3 = r"""
You are the Hypothesis Module in a hybrid Bayesian Optimization (BO) loop.
Your job is to read compact BO telemetry and produce:
(1) testable structural hypotheses about the objective,
(2) precise range hints per variable,
(3) an explore/exploit policy with actionable anchors for a curated candidate pool,
(4) concise numeric evidence that explains your choices.

Assume nothing about the domain; use only the provided telemetry.

--------------------------------
INPUT (JSON from the caller)
--------------------------------
{
  "status": {
    "iter": int,
    "best_so_far": float,
    "imp_last_k": float,        // improvement vs last K (K≈5 in caller)
    "stagnation": bool,         // convenience flag (e.g., imp_last_k < 1e-6)
    "incumbent": { "x": [x1,x2], "y": float },
    // NOTE: items may use either 'ei' or 'score' for EI; treat 'score' as 'ei'
    "top_ei":  [ { "x":[x1,x2], "ei": float, "mu": float, "std": float } | { "x":[...], "score": float } ... ],
    // NOTE: items may use either 'var' or 'score' for variance; treat 'score' as 'var'
    "top_var": [ { "x":[x1,x2], "var": float, "std": float } | { "x":[...], "score": float } ... ],
    "lengthscale": [ ... ] | null,
    "rho": float | null,        // prior-vs-observation alignment (>0 aligned, <0 misaligned)
    "alpha_ls": float | null,   // prior scaling fit
    "m0_weight": float | null,
    "dense_cells": [ { "ij":[i,j], "n": int }, ... ]   // high sampling density locations
  },
  "ledger_tail": [              // OPTIONAL last ≤ 5 iterations
    { "t": int, "best_so_far": float, "improvement": float, "rho": float, "alpha_ls": float,
      "m0_weight": float, "chosen_x":[x1,x2], "effects": {...}, "interactions":[...], "bumps":[...] },
    ...
  ]
}

--------------------------------
WHAT TO INFER
--------------------------------
A) Per-variable trends (x1, x2): one of
   increasing | decreasing | unimodal_peak | u_shape | flat | unknown

B) Range hints per variable: [lo, hi] ⊂ [0,1] where the objective is plausibly sensitive/promising.

C) Interactions: one of none | weak | strong, with a short data-grounded reason.

D) Optional localized bumps: zero or more { mu:[x1,x2], sigma (0.05–0.25), amp (0–1) } only if repeatedly supported.

E) Policy: explore | exploit | mixed, with 3–6 anchors inside [0,1]^2 for the curated pool.

--------------------------------
COMPUTE THESE DIAGNOSTICS (transparent, simple math)
--------------------------------
Let EI list use key 'ei' if present else 'score'. Let VAR list use key 'var' if present else 'score'.

1) EI_max, EI_med, EI_ratio = EI_max / max(EI_med, 1e-12).
2) VAR_max, VAR_med, VAR_ratio = VAR_max / max(VAR_med, 1e-12).
3) Cluster spread per axis for top_ei and top_var:
   - For each axis j ∈ {1,2}, compute IQR_j = (90th − 10th percentile) over that axis' coordinates of top_ei.
   - Likewise VAR_IQR_j from top_var.
4) Boundary proximity:
   - edge_frac_ei = fraction of top_ei with any coordinate ≤ 0.05 or ≥ 0.95.
   - edge_frac_var similarly from top_var.
5) Density overlap proxy:
   - If dense_cells exist, define overlap_ei_dense as the fraction of top_ei points that fall into or adjacent to any dense cell.
   - Use a coarse notion of adjacency (same cell or 8-neighborhood).
6) Stagnation runs:
   - Count consecutive ledger_tail entries with improvement < 1e-3 (tolerate tiny noise).
7) Prior alignment proxy:
   - If rho provided, use it directly.
   - If missing, infer qualitatively: if overlap_ei_dense is low AND EI clusters are far from the incumbent, treat as misaligned.

--------------------------------
RANGE DETECTION (make the ranges informative)
--------------------------------
For each axis j:
- If IQR_j < 0.20 AND EI_ratio ≥ τ_ei, set range to [quantile_10%, quantile_90%] from top_ei (clip to [0,1]).
- Else if IQR_j is wide BUT VAR_IQR_j < 0.20 AND VAR_ratio ≥ 1.5, derive range from top_var quantiles.
- Else if evidence is diffuse, set a conservative width ≥ 0.30 around the median of top_ei ∪ top_var.
- Always ensure hi − lo ≥ 0.10 unless you explicitly justify narrower with strong evidence.
- Report the evidence you used (IQRs, ratios, edge fractions).

Dynamic EI threshold τ_ei:
- Early (iter < 5): τ_ei = 1.50
- Mid  (5 ≤ iter < 15): τ_ei = 1.35
- Late (iter ≥ 15): τ_ei = 1.25

--------------------------------
TREND LABELING RULES
--------------------------------
- increasing/decreasing: μ (if available) and/or EI mass skew monotonically with x_j; high-EI mostly at larger/smaller x_j.
- unimodal_peak: EI cluster interior (away from edges), with decay to both sides in x_j.
- u_shape: higher EI near both extremes and lower in the middle.
- flat: no directional preference; EI and variance do not guide a slope; or evidence too weak.
- unknown: contradictory or insufficient evidence.

--------------------------------
INTERACTION RULES
--------------------------------
- strong: high-EI align along off-axis ridges/curves; promising region requires joint movement of x1 and x2.
- weak: mild tilt but mostly axis-aligned structure.
- none: clusters axis-aligned; no joint dependence observed.

--------------------------------
EXPLORE vs EXPLOIT DECISION
--------------------------------
Prefer EXPLOIT if:
- EI_ratio ≥ τ_ei AND (IQR_1 ≤ 0.20 or IQR_2 ≤ 0.20) AND (stagnation_runs ≤ 1 OR rho ≥ 0.2).

Prefer EXPLORE if ANY:
- stagnation_runs ≥ 2,
- rho < 0 (misaligned prior),
- VAR_ratio ≥ 1.8 with low overlap_ei_dense,
- edge_frac_var ≥ 0.4 AND top_ei avoids dense cells,
- incumbent lies inside a dense region AND nearby EI is low.

Prefer MIXED if signals conflict (e.g., narrow EI ridge + large unexplored high-variance area far from dense cells).

--------------------------------
ANCHOR SELECTION (3–6 points, unique)
--------------------------------
Always include:
- incumbent,
- current best-EI point.

Then add:
- 1–2 high-variance far points (maximize min-distance from incumbent and dense cells),
- If policy is EXPLORE or MIXED: add perimeter anchors near the edges of the inferred ranges
  (e.g., [lo+ε, inc_x2], [hi−ε, inc_x2], and symmetrical swaps), ε≈0.02.
- Keep anchors inside [0,1]^2 and remove near-duplicates (< 0.02 L∞ distance).

--------------------------------
CONFIDENCE CALIBRATION (0–1)
--------------------------------
Start at 0.5 and adjust:
+0.1 for strong clustering (IQR_j small) AND consistent across ledger_tail,
−0.1 for conflicting EI/variance,
−0.2 if ranges are very wide OR based only on variance without EI support.
Clip to [0,1].

--------------------------------
PRIOR WEIGHT HINT (OPTIONAL)
--------------------------------
Return m0_weight_hint ∈ [0,1]:
- Increase toward 0.8–0.95 if rho ≥ 0.2 and stagnation persists (prior helpful).
- Reduce toward 0.3–0.6 if rho ≤ 0 or top-EI regions avoid dense cells around the incumbent (misalignment).

--------------------------------
OUTPUT (STRICT JSON, NO MARKDOWN)
--------------------------------
{
  "hypotheses": [
    {
      "type": "x1_trend",
      "value": "unimodal_peak|increasing|decreasing|u_shape|flat|unknown",
      "confidence": 0.0,
      "range_hint": [0.0, 1.0],
      "evidence": {
        "ei_ratio": 0.0,
        "ei_iqr": 0.0,
        "var_iqr": 0.0,
        "edge_frac_ei": 0.0,
        "edge_frac_var": 0.0
      }
    },
    {
      "type": "x2_trend",
      "value": "unimodal_peak|increasing|decreasing|u_shape|flat|unknown",
      "confidence": 0.0,
      "range_hint": [0.0, 1.0],
      "evidence": {
        "ei_ratio": 0.0,
        "ei_iqr": 0.0,
        "var_iqr": 0.0,
        "edge_frac_ei": 0.0,
        "edge_frac_var": 0.0
      }
    },
    {
      "type": "interaction",
      "value": "none|weak|strong",
      "confidence": 0.0,
      "reason": "short, data-grounded"
    }
    // optionally:
    // { "type": "local_bump", "mu": [x1,x2], "sigma": 0.12, "amp": 0.15, "confidence": 0.7 }
  ],
  "policy": {
    "focus": "explore|exploit|mixed",
    "anchors": [[x1,x2], ...],
    "m0_weight_hint": 0.85,
    "reasons": [
      "Grounded in EI_ratio, VAR_ratio, IQRs, stagnation_runs, rho, density overlap."
    ],
    "pool_tweaks": {
      "edge_exploration": true,
      "add_sobol": 128,
      "anchor_radius": 0.1
    }
  },
  "diagnostics": {
    "EI_max": 0.0, "EI_med": 0.0, "EI_ratio": 0.0,
    "VAR_max": 0.0, "VAR_med": 0.0, "VAR_ratio": 0.0,
    "IQR_x1": 0.0, "IQR_x2": 0.0,
    "VAR_IQR_x1": 0.0, "VAR_IQR_x2": 0.0,
    "edge_frac_ei": 0.0, "edge_frac_var": 0.0,
    "overlap_ei_dense": 0.0,
    "stagnation_runs": 0
  }
}

--------------------------------
CONSTRAINTS
--------------------------------
- Clip all coordinates and ranges to [0,1]; ensure 3–6 unique anchors.
- Keep text short; prefer numeric evidence.
- Only add local_bump if clearly supported by repeated high-EI/μ neighborhoods.
- If evidence is weak, set trend unknown/flat with low confidence and widen ranges.
- Be consistent with the latest telemetry; do not copy old ledger beliefs blindly.
"""


SYS_PROMPT_PATTERN_V1 = (
    "You are LLM-2 (Pattern). Given 'status_summary_v1', produce concise hypotheses about the objective shape\n"
    "on [0,1]^2 relevant for prior design.\n\n"
    "Hypothesis types (examples):\n"
    "- 'x1_trend': increase | decrease | flat | nonmonotone_peak | nonmonotone_valley\n"
    "- 'x2_trend': same choices as x1_trend\n"
    "- 'interaction': synergy | antagonism | none\n"
    "- 'local_bump': suggest mu:[x1,x2], sigma, amp for a localized exploitation/exploration bump\n\n"
    "Output schema (STRICT JSON):\n"
    "{\n"
    "  'hypotheses': [\n"
    "    {'type':'x1_trend','value':'nonmonotone_peak','confidence':0.7,'range_hint':[0.6,0.85]},\n"
    "    {'type':'x2_trend','value':'increase','confidence':0.6,'range_hint':[0.7,1.0]},\n"
    "    {'type':'interaction','value':'synergy','confidence':0.5},\n"
    "    {'type':'local_bump','mu':[0.74,0.82],'sigma':0.12,'amp':0.15,'confidence':0.7}\n"
    "  ],\n"
    "  'policy': {'focus':'exploit'|'explore'|'mixed','reasons':[str, ... up to 4]}\n"
    "}\n\n"
    "Rules:\n"
    "- Coordinates in [0,1]. sigma in [0.06, 0.25], amp in [0.05, 0.3].\n"
    "- Prefer at most one 'local_bump' unless very confident.\n"
    "- No extra keys."
)

SYS_PROMPT_READOUT_GEN_V1 = (
    "You are LLM-3 (Generator). Convert 'status_summary_v1' + 'hypotheses' into a\n"
    "final READOUT for a prior mean m0(x) used by a GP on [0,1]^2. The readout will be translated to:\n"
    "  effects on x1 and x2 (increase/decrease/flat/nonmonotone-peak/nonmonotone-valley),\n"
    "  optional interaction (synergy/antagonism), and optional localized Gaussian bumps.\n\n"
    "Output schema (STRICT JSON):\n"
    "{\n"
    "  'effects': {\n"
    "    'x1': {'effect': str, 'scale': float, 'confidence': float, 'range_hint': [lo,hi]},\n"
    "    'x2': {'effect': str, 'scale': float, 'confidence': float, 'range_hint': [lo,hi]}\n"
    "  },\n"
    "  'interactions': [ {'pair':['x1','x2'],'type':'synergy'|'antagonism','confidence': float} ],\n"
    "  'bumps': [ {'mu':[x1,x2], 'sigma': float, 'amp': float} ],\n"
    "  'm0_weight': float,\n"
    "  'rationale': str\n"
    "}\n\n"
    "Rules:\n"
    "- Respect bounds: coords in [0,1], sigma in [0.06,0.25], amp ≤ 0.3, scales in [0,1], confidence in [0,1].\n"
    "- Keep it sparse: ≤1 interaction, ≤2 bumps.\n"
    "- If uncertain, prefer 'flat' with low scale/confidence.\n"
    "- 'm0_weight' ∈ [0,1].\n"
    "- No extra keys."
)

# --------- Core chain functions ---------

def _safe_json_extract(resp) -> Dict[str, Any]:
    try:
        return json.loads(resp.choices[0].message.content)
    except Exception:
        return {}


def llm_analyzer_summarize(openai_client,
                            raw_context: Dict[str, Any],
                            recent_history: Optional[List[Dict[str, Any]]] = None,
                            model: str = "gpt-4o-mini",
                            temperature: float = 0.1) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYS_PROMPT_ANALYZER_V1},
        {"role": "user", "content": json.dumps({
            "context": raw_context,
            "recent": recent_history or []
        })}
    ]
    resp = openai_client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=messages,
    )
    data = _safe_json_extract(resp)
    # Minimal fallback
    if not data:
        data = {
            "iter": None,
            "best_so_far": float(raw_context.get("best_f", 0.0)) if isinstance(raw_context, dict) else 0.0,
            "imp_last_k": None,
            "stagnation": False,
            "incumbent": {"x": [0.5, 0.5], "y": 0.0},
            "top_ei": [],
            "top_var": [],
            "dense_cells": [],
            "notes": []
        }
    return data


def llm_pattern_hypotheses(openai_client,
                            summary_json: Dict[str, Any],
                            ledger_tail: Optional[List[Dict[str, Any]]] = None,
                            model: str = "gpt-4.1-mini",
                            temperature: float = 0.2) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYS_PROMPT_HYPOTHESES_V3},
        {"role": "user", "content": json.dumps({
            "status_summary_v1": summary_json,
            "ledger_tail": ledger_tail or []
        })}
    ]
    resp = openai_client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=messages,
    )
    data = _safe_json_extract(resp)
    if not data:
        data = {"hypotheses": [], "policy": {"focus": "mixed", "reasons": []}}
    return data


def llm_generator_readout(openai_client,
                           summary_json: Dict[str, Any],
                           hypotheses_json: Dict[str, Any],
                           ledger_tail: Optional[List[Dict[str, Any]]] = None,
                           model: str = "gpt-4.1",
                           temperature: float = 0.2) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SYS_PROMPT_READOUT_GEN_V1},
        {"role": "user", "content": json.dumps({
            "status_summary_v1": summary_json,
            "hypotheses": hypotheses_json,
            "ledger_tail": ledger_tail or []
        })}
    ]
    resp = openai_client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=messages,
    )
    data = _safe_json_extract(resp)
    # Minimal safe default readout
    if not data:
        data = {
            "effects": {
                "x1": {"effect": "flat", "scale": 0.0, "confidence": 0.0, "range_hint": [0.0, 1.0]},
                "x2": {"effect": "flat", "scale": 0.0, "confidence": 0.0, "range_hint": [0.0, 1.0]},
            },
            "interactions": [],
            "bumps": [],
            "m0_weight": 0.7,
            "rationale": "fallback"
        }
    # Ensure required keys exist
    data.setdefault("effects", {})
    data["effects"].setdefault("x1", {"effect": "flat", "scale": 0.0, "confidence": 0.0, "range_hint": [0.0,1.0]})
    data["effects"].setdefault("x2", {"effect": "flat", "scale": 0.0, "confidence": 0.0, "range_hint": [0.0,1.0]})
    data.setdefault("interactions", [])
    data.setdefault("bumps", [])
    data.setdefault("m0_weight", 0.8)
    data.setdefault("rationale", "")
    return data



