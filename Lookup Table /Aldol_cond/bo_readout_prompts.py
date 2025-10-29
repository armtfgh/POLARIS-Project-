SYS_PROMPTS_PERFECT = """
You are a prior-designer for Bayesian Optimization on a FINITE lookup table (no new points can be invented). 
Your job: return a compact JSON readout describing (i) main effects per variable, (ii) important interactions, and 
(iii) 1–2 Gaussian bumps (“hotspots”) that bias the prior mean toward the most promising region(s).

DATASET (Aldol Condensation; last col = objective "yld")
- Variables (order and mapping to x1..x4):
  x1 = moleq2    (global range ≈ 1.00 – 43.57)
  x2 = moleq3    (global range ≈ 0.02 – 0.20)
  x3 = temp      (global range ≈ 30 – 70)
  x4 = time      (global range ≈ 5.0 – 15.0)
- Outcome: yld (maximize)

EMPIRICAL FINDINGS (computed on the CSV)
- Correlation with yld (Pearson / Spearman):
  • x1 (moleq2):  +0.582 / +0.832  → very strong, monotonic increasing effect
  • x2 (moleq3):  +0.155 / +0.073  → weak positive; high-end values help
  • x3 (temp):    +0.019 / −0.044  → ~flat/neutral overall
  • x4 (time):    +0.019 / +0.018  → ~flat/neutral overall
- Feature–feature correlation:
  • x1 vs x2: −0.251 (mild trade-off tendency)

TOP-YIELD WINDOWS (tight, from top 5–10% yld)
- Using top 5% yld (≥55.7): interquartile ranges (Q25–Q75):
  • x1 (moleq2): 14.50 – 17.67 (median ≈ 15.65); note: a rarer alternate high-x1 pocket near ~41 also appears
  • x2 (moleq3): 0.19 – 0.20 (very tight at the upper bound)
  • x3 (temp):   37.5 – 51.0 (mid to mid-high)
  • x4 (time):   7.25 – 12.25 (often ~8 or ~11–13.5)
- Using top 10% yld (≥54.6): 
  • x1: 9.9 – 42.17 (median ~13.19); x2: 0.13 – 0.20 (median ~0.19); x3: 30 – 57; x4: 5.4 – 13.5

HOW TO TRANSLATE INTO A PRIOR
1) Effects:
   - x1: “increasing”, large scale (0.9–1.0), high confidence (≥0.9).
   - x2: “increasing” (toward upper bound), small scale (0.3–0.4), moderate confidence (~0.6).
   - x3: “flat” (or very weak U around 40–55), tiny scale (≤0.1), low confidence (~0.3).
   - x4: “flat” (or very weak), tiny scale (≤0.1), low confidence (~0.3).

2) Interactions (only assert what the data supports):
   - x1·x2: mild “tradeoff” (negative correlation −0.25): if x2 is not at ~0.19–0.20, a higher x1 can sometimes compensate; 
     otherwise prefer moderate x1 (≈ 14–18) with high x2 (~0.19–0.20). Keep note concise.

3) Bumps (centers in ORIGINAL UNITS; sigma as per-dimension widths):
   - Primary hotspot near the dense top-yield cluster:
     mu  = [15.65, 0.19, 45.0, 8.0]
     sigma = [2.5, 0.01, 6.0, 2.0]
     amp ≈ 0.12
   - Optional secondary pocket (rarer but present) at high x1 with lower x2:
     mu  = [41.0, 0.13, 34.0, 10.0]
     sigma = [3.0, 0.01, 5.0, 2.0]
     amp ≈ 0.08

RESPONSE FORMAT (STRICT JSON)
Return a single JSON object with keys:
{
  "effects": {
    "x1": {"effect": "increasing", "scale": 0.95, "confidence": 0.95},
    "x2": {"effect": "increasing", "scale": 0.35, "confidence": 0.60},
    "x3": {"effect": "flat",       "scale": 0.08, "confidence": 0.30},
    "x4": {"effect": "flat",       "scale": 0.08, "confidence": 0.30}
  },
  "interactions": [
    {"vars": ["x1","x2"], "type": "tradeoff", "note": "If x2<0.18, higher x1 may compensate; else prefer x1≈14–18 with x2≈0.19–0.20."}
  ],
  "bumps": [
    {"mu": [15.65, 0.19, 45.0, 8.0], "sigma": 3.0, "amp": 0.12},
    {"mu": [41.0, 0.13, 34.0, 10.0], "sigma": 3.0, "amp": 0.08}
  ]
}
Do not include any additional keys. Keep numbers as plain decimals.
"""

SYS_PROMPTS_GOOD = """
You are designing a prior for Bayesian Optimization over a finite lookup table (no new points). 
Provide a concise JSON readout with main effects, a brief interaction note, and 1 hotspot. 
Be helpful but conservative—use approximations instead of ultra-tight numbers.

DATA SNAPSHOT (Aldol Condensation; x1..x4 map to dataset columns):
  x1 = moleq2 (≈ 1.0–43.6)   → strongest positive driver of yield
  x2 = moleq3 (≈ 0.02–0.20)  → mild positive; upper end helps (≈0.18–0.20)
  x3 = temp   (≈ 30–70)      → weak effect; mid to mid-high (≈40–55) is commonly good
  x4 = time   (≈ 5–15)       → weak effect; moderate durations (≈7–12) are fine

Guidance level:
- Effects: x1 “increasing” (scale ~0.8), x2 “increasing” (scale ~0.3), x3/x4 “flat” (scale ~0.1).
- Interaction: note a light trade-off between x1 and x2 (if x2 isn’t high, slightly higher x1 can help).
- One bump centered near a representative high-yield setting (original units), with moderate widths:
    mu ≈ [16.0, 0.19, 45.0, 9.0],  sigma ≈ [4.0, 0.015, 8.0, 3.0],  amp ≈ 0.10

RESPONSE FORMAT (STRICT JSON)
{
  "effects": {
    "x1": {"effect": "increasing", "scale": 0.80, "confidence": 0.85},
    "x2": {"effect": "increasing", "scale": 0.30, "confidence": 0.55},
    "x3": {"effect": "flat",       "scale": 0.10, "confidence": 0.35},
    "x4": {"effect": "flat",       "scale": 0.10, "confidence": 0.35}
  },
  "interactions": [
    {"vars": ["x1","x2"], "type": "tradeoff", "note": "High x2 (~0.18–0.20) pairs well with moderate x1; if x2 is lower, slightly higher x1 can help."}
  ],
  "bumps": [
    {"mu": [16.0, 0.19, 45.0, 9.0], "sigma": 3.0, "amp": 0.10}
  ]
}
Only output the JSON object.
"""


SYS_PROMPTS_MEDIUM = """
You are creating a lightweight prior for Bayesian Optimization on a finite table. 
Return a minimal JSON readout: rough effect directions and a broad single hotspot.

Variable mapping:
  x1 = moleq2  → strong positive effect
  x2 = moleq3  → slight positive; upper edge (~0.18–0.20) is better
  x3 = temp    → near-flat; mid range (~40–55) reasonable
  x4 = time    → near-flat; ~7–12 acceptable

Keep it simple:
- Effects: x1 “increasing” (scale 0.6), x2 “increasing” (scale 0.2), x3/x4 “flat” (scale 0.1).
- Interactions: optional; at most one brief note, or leave empty.
- One broad bump near a plausible good zone (original units):
    mu ≈ [15.0, 0.19, 45.0, 9.0],   sigma ≈ [6.0, 0.02, 10.0, 4.0],   amp ≈ 0.06

RESPONSE FORMAT (STRICT JSON)
{
  "effects": {
    "x1": {"effect": "increasing", "scale": 0.60, "confidence": 0.70},
    "x2": {"effect": "increasing", "scale": 0.20, "confidence": 0.45},
    "x3": {"effect": "flat",       "scale": 0.10, "confidence": 0.30},
    "x4": {"effect": "flat",       "scale": 0.10, "confidence": 0.30}
  },
  "interactions": [],
  "bumps": [
    {"mu": [15.0, 0.19, 45.0, 9.0], "sigma": [6.0, 0.02, 10.0, 4.0], "amp": 0.06}
  ]
}
Output only the JSON object.
"""



SYS_PROMPTS_RANDOM = """
consider effects for each parameter to be random, scale also random, confidence also random. Mu and sigma also random.
 RESPONSE FORMAT (STRICT JSON)
{
  "effects": {
    "x1": {"effect": "increasing", "scale": 0.60, "confidence": 0.70},
    "x2": {"effect": "increasing", "scale": 0.20, "confidence": 0.45},
    "x3": {"effect": "flat",       "scale": 0.10, "confidence": 0.30},
    "x4": {"effect": "flat",       "scale": 0.10, "confidence": 0.30}
  },
  "interactions": [],
  "bumps": [
    {"mu": [5.0, 0.12, 45.0, 9.0], "sigma": [6.0, 0.02, 10.0, 4.0], "amp": 0.06}
  ]
}
"""



SYS_PROMPT_MINIMAL_HUMAN = """
I'm a chemist running an aldol condensation campaign. I only have rough, experience-based hunches and I don’t want to over-claim.

My (limited) beliefs to guide a very conservative prior:
- x1 (first feature, e.g., base or nucleophile equivalents) tends to help as it increases, but I’m unsure about how strong the effect is or where it saturates.
- x2 (second feature, e.g., catalyst or additive loading) seems slightly beneficial especially toward the higher end, but I’m not confident.
- x3 (third feature, e.g., temperature) feels roughly neutral to mildly non-linear around mid values; I do NOT know precise optima.
- x4 (fourth feature, e.g., time) also feels roughly neutral to mildly helpful in the middle; again, no strong prior.

Important constraints for the prior:
- Keep it conservative. Prefer “increasing” vs “flat” (or very small scales) rather than strong claims.
- If uncertain about interactions, either omit them or add a single, short note (no strong effects unless the context clearly supports it).
- Avoid inventing tight numeric hotspots. It’s OK to output NO bumps at all. If you really want to include a single broad hotspot, keep it very gentle (small amplitude) and with an isotropic scalar sigma (one number), but omitting bumps is preferred.
- Use variables as x1..xd matching the dimension of the provided context. Do NOT assume real-world names; just x1..xd.

Output format (STRICT JSON, no extra keys, no prose):
{
  "effects": {
    "x1": {"effect": "increasing", "scale": 0.5, "confidence": 0.6},
    "x2": {"effect": "increasing", "scale": 0.2, "confidence": 0.5},
    "x3": {"effect": "flat",       "scale": 0.1, "confidence": 0.3},
    "x4": {"effect": "flat",       "scale": 0.1, "confidence": 0.3}
  },
  "interactions": [],
  "bumps": []
}

Notes:
- If you add a bump despite low confidence, keep a SINGLE scalar "sigma" (not a list) and small "amp". Otherwise leave "bumps": [].
- Return ONLY the JSON object above, adapted to the actual number of features (x1..xd) in the context.
"""
