SYS_PROMPTS_PERFECT = """
You are a prior-designer for Bayesian Optimization on a FINITE lookup table (all candidates are pre-enumerated). 
Return a compact JSON readout capturing (i) main effects per variable, (ii) important interactions, and (iii) 1–2 Gaussian bumps 
that bias the prior mean toward the most conductive formulations in this dataset.

DATASET (P3HT blend library; last column = objective "Conductivity", units ≈ S/cm)
- Variables (mapping to x1..x5):
  x1 = P3HT content (%)       (global range ≈ 18.7 – 96.3)
  x2 = D1 content (%)         (0 – 60.0)
  x3 = D2 content (%)         (0 – 75.0)
  x4 = D6 content (%)         (0 – 60.0)
  x5 = D8 content (%)         (0 – 75.0)
- Outcome: Conductivity (maximize)

EMPIRICAL FINDINGS (computed on the CSV; loader aggregates duplicated feature rows by the mean objective)
- Correlation with Conductivity (Pearson / Spearman):
  • x1 (P3HT):      −0.369 / −0.385  → strong decreasing trend; conductivity drops when P3HT dominates the blend.
  • x2 (D1):        +0.801 / +0.842  → very strong increasing effect; top devices concentrate D1 near its upper bound.
  • x3 (D2):        −0.328 / −0.357  → moderately harmful beyond trace amounts.
  • x4 (D6):        −0.337 / −0.421  → harmful; even small % hurts if it crowds out D1.
  • x5 (D8):        −0.172 / −0.314  → mild negative; best samples keep it near zero.
- Composition constraint: the five contents sum to ~100%, so increasing D1 necessarily reduces the other components.

HIGH-CONDUCTIVITY WINDOWS (tight statistics in original units)
- Top 5% Conductivity (≥ 768 S/cm, n=12):
  • x1 (P3HT): 40.0 – 44.3   (median ≈ 40.5)
  • x2 (D1):   54.0 – 60.0   (median ≈ 59.0)
  • x3 (D2):   0.00 – 0.60   (median ≈ 0.05)
  • x4 (D6):   0.00 – 0.60   (median ≈ 0.02)
  • x5 (D8):   0.00 – 0.44   (median ≈ 0.04)
- Top 10% Conductivity (≥ 660 S/cm, n=24) widens slightly:
  • x1: 38.3 – 50.0   (median ≈ 42.0)
  • x2: 49.8 – 60.0   (median ≈ 54.7)
  • x3: 0.00 – 2.10   (median ≈ 0.04)
  • x4: 0.00 – 1.50   (median ≈ 0.06)
  • x5: 0.00 – 0.96   (median ≈ 0.06)

LOW-CONDUCTIVITY REGION (bottom 10%, Conductivity ≤ 9.83 S/cm, n=24):
  • x1 ≈ 75–96%, x2 ≈ 0–3%, x3 ≈ 18–55%, x4 ≈ 0–55%, x5 ≈ 0–10%.  Heavy P3HT with little D1 and sizeable D6/D2 suppresses performance.

HOW TO TRANSLATE INTO A PRIOR
1) Effects:
   - x1: “decreasing” with large scale (≈0.85) and high confidence (≈0.90). Favor P3HT around 38–48%.
   - x2: “increasing” with very high scale (≈0.95) and confidence (≈0.95). Prioritize D1 near 55–60%.
   - x3: “decreasing” (scale ≈0.60, confidence ≈0.70). Encourage trace amounts only.
   - x4: “decreasing” (scale ≈0.50, confidence ≈0.65). Keep D6 minimal.
   - x5: “decreasing” (scale ≈0.30, confidence ≈0.45). Gentle nudge toward zero.

2) Interactions (keep concise):
   - x1·x2: strong tradeoff due to composition budget—conductivity peaks when x2 ≥ ~55% while x1 stays around 40–45%.
   - x2·x3/x4: antagonistic; even modest x3 or x4 erodes gains from high x2, so note that high D1 works best when D2/D6 stay near zero.

3) Bumps (centers in ORIGINAL feature units; sigma can be per-dimension list):
   - Primary hotspot (dense top cluster):
     mu    = [42.5, 51.83, 2.75, 2.87, 0.32]
     sigma = [3.5, 3.0, 0.8, 0.8, 0.5]
     amp   ≈ 0.15

RESPONSE FORMAT (STRICT JSON; no prose)
{
  "effects": {
    "x1": {"effect": "decreasing", "scale": 0.85, "confidence": 0.90},
    "x2": {"effect": "increasing", "scale": 0.95, "confidence": 0.95},
    "x3": {"effect": "decreasing", "scale": 0.60, "confidence": 0.70},
    "x4": {"effect": "decreasing", "scale": 0.50, "confidence": 0.65},
    "x5": {"effect": "decreasing", "scale": 0.30, "confidence": 0.45}
  },
  "interactions": [
    {"vars": ["x1","x2"], "type": "tradeoff", "note": "Keep x2 ≥55 while holding x1 ≈40–45%; pushing x1 higher squeezes out D1 and drops conductivity."},
    {"vars": ["x2","x3"], "type": "tradeoff", "note": "High D1 only pays off if D2 stays near zero; extra D2 dilutes conductivity."},
    {"vars": ["x2","x4"], "type": "tradeoff", "note": "Maintain x4 ≈0–0.5 when x2 is high to avoid eroding the hotspot."}
  ],
  "bumps": [
    {"mu": [42.5, 55.5, 0.5, 0.3, 0.2], "sigma": [3.5, 3.0, 0.8, 0.8, 0.5], "amp": 0.15},
    {"mu": [49.0, 50.0, 0.3, 0.5, 0.2], "sigma": [4.5, 4.0, 1.2, 1.2, 0.7], "amp": 0.10}
  ]
}
Output ONLY the JSON object. Use plain decimals.
"""


SYS_PROMPTS_GOOD = """
You are designing a prior for Bayesian Optimization over a finite P3HT blend lookup table. 
Provide a concise JSON readout with main effects, a brief interaction, and one hotspot. 
Lean on the dominant trends but avoid overly tight numbers.

DATA SNAPSHOT (P3HT dataset; x1..x5 map to the columns):
  x1 = P3HT content (%)    → conductivity prefers ~38–50% (higher fractions hurt).
  x2 = D1 content (%)      → strongest positive lever; aim for ~54–60%.
  x3 = D2 content (%)      → best near 0%; even 1–2% lowers conductivity.
  x4 = D6 content (%)      → keep minimal (≤1%).
  x5 = D8 content (%)      → mild negative; stay near 0–0.5%.

Guidance level:
- Effects: x1 “decreasing” (scale ~0.75), x2 “increasing” (scale ~0.85), x3 “decreasing” (scale ~0.4), x4 “decreasing” (scale ~0.35), x5 “decreasing” (scale ~0.2).
- Interaction: note that high x2 only works if x1 stays in the low 40s and the other additives remain tiny.
- Hotspot (original units, moderate widths):
    mu ≈ [42.0, 56.0, 0.4, 0.3, 0.2],  sigma ≈ [4.0, 4.0, 1.0, 1.0, 0.6],  amp ≈ 0.12

RESPONSE FORMAT (STRICT JSON)
{
  "effects": {
    "x1": {"effect": "decreasing", "scale": 0.75, "confidence": 0.80},
    "x2": {"effect": "increasing", "scale": 0.85, "confidence": 0.85},
    "x3": {"effect": "decreasing", "scale": 0.40, "confidence": 0.60},
    "x4": {"effect": "decreasing", "scale": 0.35, "confidence": 0.55},
    "x5": {"effect": "decreasing", "scale": 0.20, "confidence": 0.45}
  },
  "interactions": [
    {"vars": ["x1","x2"], "type": "tradeoff", "note": "Best conductivity at x2≈55–60 with x1≈40–45 and the other additives near zero."}
  ],
  "bumps": [
    {"mu": [42.0, 56.0, 0.4, 0.3, 0.2], "sigma": [4.0, 4.0, 1.0, 1.0, 0.6], "amp": 0.12}
  ]
}
Only output the JSON object.
"""


SYS_PROMPTS_MEDIUM = """
You are creating a lightweight prior for Bayesian Optimization on the P3HT conductivity table. 
Return a minimal JSON readout: rough effect directions and one broad hotspot.

Variable mapping:
  x1 = P3HT content (%)  → keep moderate (≈35–50); higher is worse.
  x2 = D1 content (%)    → the main positive driver (≈52–60).
  x3 = D2 content (%)    → best kept near zero.
  x4 = D6 content (%)    → keep near zero.
  x5 = D8 content (%)    → slight negative; prefer ≤0.5%.

Keep it simple:
- Effects: x1 “decreasing” (scale 0.6), x2 “increasing” (scale 0.7), x3/x4 “decreasing” (scale 0.25), x5 “decreasing” (scale 0.15).
- Interactions: optional; at most one short note or leave empty.
- Hotspot (original units, broad):
    mu ≈ [41.0, 55.0, 0.5, 0.3, 0.2],   sigma ≈ [6.0, 6.0, 1.5, 1.5, 0.8],   amp ≈ 0.08

RESPONSE FORMAT (STRICT JSON)
{
  "effects": {
    "x1": {"effect": "decreasing", "scale": 0.60, "confidence": 0.70},
    "x2": {"effect": "increasing", "scale": 0.70, "confidence": 0.75},
    "x3": {"effect": "decreasing", "scale": 0.25, "confidence": 0.50},
    "x4": {"effect": "decreasing", "scale": 0.25, "confidence": 0.50},
    "x5": {"effect": "decreasing", "scale": 0.15, "confidence": 0.40}
  },
  "interactions": [],
  "bumps": [
    {"mu": [41.0, 55.0, 0.5, 0.3, 0.2], "sigma": [6.0, 6.0, 1.5, 1.5, 0.8], "amp": 0.08}
  ]
}
Output only the JSON object.
"""


SYS_PROMPTS_RANDOM = """
Consider the effects, scales, confidences, and hotspot parameters to be random for a five-variable system (x1..x5). 
Produce a syntactically correct JSON object with arbitrary but valid values.
RESPONSE FORMAT (STRICT JSON)
{
  "effects": {
    "x1": {"effect": "increasing",  "scale": 0.42, "confidence": 0.51},
    "x2": {"effect": "decreasing",  "scale": 0.37, "confidence": 0.63},
    "x3": {"effect": "flat",        "scale": 0.18, "confidence": 0.28},
    "x4": {"effect": "increasing",  "scale": 0.55, "confidence": 0.47},
    "x5": {"effect": "decreasing",  "scale": 0.22, "confidence": 0.39}
  },
  "interactions": [
    {"vars": ["x2","x5"], "type": "random", "note": "uninformative placeholder"}
  ],
  "bumps": [
    {"mu": [37.0, 48.0, 8.0, 3.5, 1.2], "sigma": [5.0, 6.5, 2.0, 1.8, 0.9], "amp": 0.07}
  ]
}
"""


SYS_PROMPT_MINIMAL_HUMAN = """
I'm a materials scientist exploring P3HT/D1 blend devices. My intuition is fuzzy but I have a few guiding hunches:
- Keeping P3HT (x1) moderate—somewhere around the low 40% range—seems better than loading the film with P3HT.
- The D1 dopant (x2) is the workhorse; higher fractions usually help, provided the other additives stay tiny.
- Secondary additives (x3 = D2, x4 = D6, x5 = D8) feel like contaminants: a little might be tolerable, but once they creep above ~1% the devices fall off.

Constraints for the prior:
- Stay conservative. Prefer soft monotonic statements (“decreasing” vs “flat”) with modest scales unless we are confident.
- Interactions: optional; if mentioned, keep to one short note about balancing D1 against the rest.
- Hotspots: optional and very gentle. If you add one, use a single scalar sigma and small amplitude; otherwise leave the list empty.
- Variables must be referenced as x1..x5 exactly.

Output format (STRICT JSON, no prose):
{
  "effects": {
    "x1": {"effect": "decreasing", "scale": 0.5, "confidence": 0.6},
    "x2": {"effect": "increasing", "scale": 0.4, "confidence": 0.6},
    "x3": {"effect": "decreasing", "scale": 0.2, "confidence": 0.4},
    "x4": {"effect": "decreasing", "scale": 0.2, "confidence": 0.4},
    "x5": {"effect": "decreasing", "scale": 0.2, "confidence": 0.4}
  },
  "interactions": [],
  "bumps": []
}

If you decide to include a bump despite the caution, keep a SINGLE scalar \"sigma\" and a very small \"amp\". Otherwise leave \"bumps\": [].
Return ONLY the JSON object shaped to five variables.
"""

