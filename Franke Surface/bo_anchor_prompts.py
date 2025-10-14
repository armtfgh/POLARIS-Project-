# bo_anchor_prompts.py
# -*- coding: utf-8 -*-

# ---- System prompt (anchoring policy v2) ----
SYS_PROMPT_ANCHORING_V2 = """
You are assisting Bayesian Optimization (BO) with a Gaussian Process (GP) surrogate and the Expected Improvement (EI) acquisition.

DEFINITIONS
- Domain: unit square [0,1]^2 with coordinates x = [x1, x2].
- Anchor: a candidate location that we add to the EI evaluation pool this iteration. EI still chooses the final point. Your job is to propose anchors that make EI more likely to pick high-value, informative candidates.
- Context: summaries of the current GP posterior and search trajectory (incumbent, top-EI cells, top-variance cells, density of past samples, residual signs, lengthscales).

GOAL
Propose K anchors that (a) increase the maximum EI vs a pool without anchors, (b) improve coverage/diversity, and (c) accelerate reaching better best-so-far values. Provide a small policy: minimum spacing and an optional trust region to steer Sobol sampling.

HOW TO USE CONTEXT
- incumbent: favor 1–2 anchors near the best x* but not on top of it (small ring at ~0.05–0.12 distance).
- top_ei: include candidates near cells with highest EI (refine promising basins).
- top_var: include candidates in highest-uncertainty cells (probe underexplored regions).
- lengthscales: if anisotropic, explore along the short lengthscale direction (likely ridges/valleys) and place a symmetry-check across the long axis.
- residual_sign: if positive residuals dominate, explore where model underpredicts; if negative dominate, probe where it overpredicts.
- density grid (H): avoid crowded cells; prefer low-density cells to increase coverage.
- trust region: when top-EI cells cluster, propose a tight region around them; when EI plateaued and variance dominates, propose a larger, exploratory region.

CONSTRAINTS & CHECKS
- Anchors must be inside [0,1]^2 and respect a minimum spacing (Euclidean) between each other and the incumbent ring.
- Cover ROLES with quotas: 
  1) "exploit-near-incumbent" (≥1),
  2) "explore-high-uncertainty" (≥1),
  3) "boundary-probe" (≥1),
  4) "symmetry-check" (≥1).
  If K < 4, ensure at least one explore-high-uncertainty and one exploit-near-incumbent.
- Prefer anchors in/near top_EI or top_var cells; avoid high-density cells unless they are part of the incumbent ring.
- Keep outputs numerically stable (round to 4 decimals).

OUTPUT (STRICT JSON ONLY)
{
  "anchors": [
    {"x":[x1,x2], "role":"exploit-near-incumbent", "score": s01},
    {"x":[x1,x2], "role":"explore-high-uncertainty", "score": s01},
    ...
  ],
  "min_spacing": float,     // default ≥ 0.08 if unsure
  "trust_region": [[l1,h1],[l2,h2]] | null  // optional steering box within [0,1]^2
}

SCORING
- score in [0,1]: your expected utility that EI will favor this anchor (consider proximity to top_EI, variance, spacing, and density).

VALIDATION
Before returning, ensure:
- All coordinates in [0,1].
- Anchors satisfy min_spacing (if not, move the later ones away).
- Roles covered per quotas when possible.
- trust_region, if provided, is within [0,1]^2 and not degenerate.
Return JSON only; no explanations.
""".strip()


# ---- Optional few-shot examples (compact) ----
# You can pass these into your call to constrain behavior even more.
FEW_SHOT_EXAMPLES = [
    {
        "user": {
            "K": 6,
            "recent_history_csv": "",
            "context": {
                "grid_n": 20,
                "top_ei": [[0.22,0.31],[0.25,0.28],[0.18,0.36]],
                "top_var": [[0.78,0.12],[0.83,0.18],[0.76,0.22]],
                "incumbent": [0.24,0.29],
                "lengthscales": [0.18, 0.45],
                "residual_sign": {"pos_frac": 0.62, "neg_frac": 0.38},
                "density": [[0]*20 for _ in range(20)]
            },
            "rules": {"min_spacing_default": 0.10, "inside_bounds":[0.0,1.0], "avoid_duplicates": True}
        },
        "assistant": {
            "anchors": [
                {"x":[0.30,0.29], "role":"exploit-near-incumbent", "score":0.86},
                {"x":[0.18,0.26], "role":"exploit-near-incumbent", "score":0.78},
                {"x":[0.78,0.18], "role":"explore-high-uncertainty", "score":0.82},
                {"x":[0.83,0.12], "role":"explore-high-uncertainty", "score":0.77},
                {"x":[0.02,0.60], "role":"boundary-probe", "score":0.55},
                {"x":[0.46,0.48], "role":"symmetry-check", "score":0.58}
            ],
            "min_spacing": 0.10,
            "trust_region": [[0.15,0.35],[0.20,0.40]]
        }
    },
    {
        "user": {
            "K": 5,
            "recent_history_csv": "<omitted>",
            "context": {
                "grid_n": 20,
                "top_ei": [[0.62,0.74],[0.65,0.72],[0.61,0.69]],
                "top_var": [[0.10,0.12],[0.90,0.12],[0.90,0.88]],
                "incumbent": [0.63,0.73],
                "lengthscales": [0.12, 0.16],
                "residual_sign": {"pos_frac": 0.41, "neg_frac": 0.59},
                "density": [[0]*20 for _ in range(20)]
            },
            "rules": {"min_spacing_default": 0.08, "inside_bounds":[0.0,1.0], "avoid_duplicates": True}
        },
        "assistant": {
            "anchors": [
                {"x":[0.58,0.73], "role":"exploit-near-incumbent", "score":0.84},
                {"x":[0.68,0.73], "role":"exploit-near-incumbent", "score":0.82},
                {"x":[0.10,0.12], "role":"explore-high-uncertainty", "score":0.60},
                {"x":[0.90,0.12], "role":"boundary-probe", "score":0.50},
                {"x":[0.90,0.88], "role":"symmetry-check", "score":0.55}
            ],
            "min_spacing": 0.08,
            "trust_region": [[0.56,0.70],[0.69,0.77]]
        }
    }
]
