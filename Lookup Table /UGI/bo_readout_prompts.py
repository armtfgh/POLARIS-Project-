PROMPT_HEADER = """
You are a prior-designer for Bayesian Optimisation on a FINITE lookup table (all candidates are enumerated).
Summarise plausible structure-activity trends as JSON: (i) main effects per variable, (ii) a handful of interactions,
and (iii) 1-2 Gaussian bumps that bias the prior mean toward promising UGI reaction mixtures.
"""

UGI_DATASET_CONTEXT = """
DATASET (four-component Ugi coupling; objective = isolated yield, fraction 0.0–0.16)
- Variables (mapping to x1..x4):
  x1 = amine_mM       (120–300 mM)  → excessive amine suppresses yield.
  x2 = aldehyde_mM    (120–300 mM)  → moderately positive lever when paired with isocyanide.
  x3 = isocyanide_mM  (120–300 mM)  → strongest positive driver; peak near the 280–300 mM edge.
  x4 = ptsa           (0.022–0.30 equiv) → Bronsted acid promoter; higher loadings (≈0.25–0.30) accelerate convergence.
- Global trend snapshot (empirical correlations):
  yield vs x1: ρ ≈ -0.20  |  yield vs x2: ρ ≈ +0.23  |  yield vs x3: ρ ≈ +0.37  |  yield vs x4: ρ ≈ +0.40.
"""

UGI_RESPONSE_FORMAT = """
RESPONSE FORMAT (STRICT JSON; no prose, and REPLACE ALL PLACEHOLDER NUMBERS WITH YOUR OWN VALUES)
{
  "effects": {
    "x1": {"effect": "<string>", "scale": <float>, "confidence": <float>, "range_hint": [<low>, <high>]},
    "x2": {"effect": "<string>", "scale": <float>, "confidence": <float>, "range_hint": [<low>, <high>]},
    "x3": {"effect": "<string>", "scale": <float>, "confidence": <float>, "range_hint": [<low>, <high>]},
    "x4": {"effect": "<string>", "scale": <float>, "confidence": <float>, "range_hint": [<low>, <high>]}
  },
  "interactions": [
    {"vars": ["x_i","x_j"], "type": "<synergy|tradeoff>", "note": "<short explanation>"}
  ],
  "bumps": [
    {"mu": [<x1_raw>, <x2_raw>, <x3_raw>, <x4_raw>], "sigma": [<sx1>, <sx2>, <sx3>, <sx4>], "amp": <float>}
  ]
}
"""


def _prompt_template(body: str) -> str:
    return f"""{PROMPT_HEADER}
{UGI_DATASET_CONTEXT}
{body.strip()}

{UGI_RESPONSE_FORMAT}
"""


SYS_PROMPTS_PERFECT = _prompt_template(
    """
Observations from the high-yield frontier (~0.14–0.16):
1) x1 (amine): keep at the minimum (≈120–150 mM). Any excess amine quenches the acid and kills yield.
2) x2 (aldehyde): monotone increasing up to ~280 mM but saturates afterwards; moderate scale.
3) x3 (isocyanide): strict increasing; the best runs pin x3 at the ceiling (≥290 mM).
4) x4 (ptsa): sharply peaked around 0.14–0.15 equiv; raising the acid toward 0.20+ lowers yield in all cases.

Interactions to codify:
- (x2, x3) synergy: both need to be high to leverage the Ugi manifold.
- (x1, x3) tradeoff: high amine erodes the benefits of excess isocyanide.
- (x3, x4) mild synergy: the acid boost matters most when x3 is high.

Hotspots (original units):
  * Primary: mu=[120, 300, 300, 0.14], sigma=[18, 12, 8, 0.008], amp=0.18
"""
)


SYS_PROMPTS_GOOD = _prompt_template(
    """
Craft an informed but slightly relaxed prior:
- Penalise x1 when it exceeds ~200 mM (use a decreasing effect with moderate scale).
- Reward x2/x3 jointly; treat x3 as the dominant increasing axis, x2 as supportive.
- Model x4 as a peaked acid window centred at ≈0.25 with gentle width.
- Include one synergy term for (x2,x3) and one tradeoff for (x1,x3).
- Provide a single bump with mu≈[150, 260, 285, 0.25] and broader sigmas (~30 mM on reagents, 0.03 on x4).
"""
)


SYS_PROMPTS_MEDIUM = _prompt_template(
    """
Produce a cautious prior rooted in the qualitative trends only:
- Note that x1 tends to hurt yield once it moves above its median; encode a mild decreasing effect.
- Treat x2 and x3 as monotone increasing but cap their confidence at 0.5.
- Let x4 be “increase then saturate” rather than a sharp peak (use increasing with range_hint).
- Mention a single interaction highlighting that x4 only helps when x3 is non-zero.
- Keep the Gaussian bump coarse (mu ~ [170, 230, 260, 0.22], sigma ~ [40, 50, 50, 0.05]).
"""
)


SYS_PROMPTS_RANDOM = _prompt_template(
    """
Deliberately produce a noisy or contradictory readout to simulate unhelpful advice.
- Assign random effect directions (use only the allowed words: increasing, decreasing, nonmonotone-peak, nonmonotone-valley, flat) irrespective of the data trends.
- Provide at least one interaction that mixes unrelated axes (e.g., x1 with x4) with a vague explanation.
- Place the Gaussian bump near the centre of the domain instead of the true optimum.
"""
)


SYS_PROMPTS_BAD = _prompt_template(
    """
Encode a confidently wrong prior to stress-test robustness:
- Claim that x1 should be maximised (“increasing”) and that low amine "starves the coupling".
- Recommend minimising x2 and x3 (“decreasing”) to "avoid side reactions".
- Push x4 toward its minimum (“decreasing”), stating that acid harms selectivity.
- Place the bump at mu≈[280, 140, 150, 0.05] with narrow sigmas.
"""
)


SYS_PROMPT_MINIMAL_HUMAN = _prompt_template(
    """
Provide a conservative, human-authored prior:
- Effects: x1 = decreasing (scale~0.3), x2 = mildly increasing, x3 = increasing, x4 = peaked around 0.25.
- Interaction: (x1,x3) tradeoff only.
- One bump near mu=[150, 260, 290, 0.25] with sigma=[25, 25, 20, 0.025], amp=0.1.
"""
)


SYS_PROMPTS_CUSTOM = _prompt_template(
    """
Focus on balanced reagent stoichiometry and ptSA titration experiments:
- Encourage an "L-shaped" manifold where x1 stays near 140–180 mM (use decreasing outside that window) while x2 and x3 co-vary along the diagonal (increasing with range hints).
- Add a secondary bump for the acid sweep: mu=[140, 240, 280, 0.23], sigma=[20, 40, 30, 0.02].
- Highlight that the acid optimum drifts lower (0.22) if x1 is not fully minimised.
"""
)


SYS_PROMPTS_CUSTOM_GOOD = SYS_PROMPTS_GOOD



SYS_PROMPTS_BEST = _prompt_template(
    """
The analysis of the UGI reaction data reveals several insights into the chemical landscape:

1. **Primary Drivers**:
   - **x2 (Aldehyde)** and **x3 (Isocyanide)** are the primary drivers of the reaction yield. Both variables exhibit positive correlations with yield, with x3 showing a stronger correlation (corr = 0.3664) than x2 (corr = 0.2363). This suggests that increasing the concentrations of aldehyde and isocyanide generally leads to higher yields.

2. **Catalyst Impact**:
   - **x4 (Catalyst)** has the highest positive correlation with yield (corr = 0.4459), indicating that the amount of catalyst is a significant factor in achieving higher yields. The data suggests that higher equivalents of catalyst are beneficial.

5. **Low-Yield Conditions**:
   - Low yields are frequently associated with low catalyst levels (x4 around 0.021931 equivalents) and lower concentrations of x2 and x3. This suggests that insufficient catalyst and lower reactant concentrations are detrimental to the reaction yield.

6. **Interactions**:
   - High yields are achieved when both x2 and x3 are high, indicating a synergistic effect between these two variables. The presence of a sufficient amount of catalyst (x4) further enhances this effect.


In summary, the reaction is primarily driven by high concentrations of aldehyde and isocyanide, with the catalyst playing a crucial role in enhancing yields. The amine concentration appears to be less influential. For optimal yields, focus on maximizing x2, x3, and x4 within their observed ranges.

"""
)



# SYS_PROMPTS_BEST = _prompt_template(
#     """
# Use the absolute best experimental run as the anchor. The highest observed yield (≈0.156) occurs for:
#     x1 ≈ 120 mM (amine), x2 ≈ 270 mM (aldehyde), x3 ≈ 300 mM (isocyanide), x4 ≈ 0.120 ptSA.

# Instructions:
# - Force extremely tight range hints around those values: x1 in [118, 125], x2 in [265, 305], x3 in [295, 300], x4 in [0.11, 0.13].
# - Treat x1 as strongly decreasing outside that window (scale≈1.0). x2 and x3 should be “increasing” with high confidence but saturate in that precise range. x4 must be “nonmonotone-peak” centred at 0.120 with sigma ≈0.005.
# - Add a single Gaussian bump exactly at [120, 270, 300, 0.120] with sigma matching those narrow windows.
# - Interactions: emphasise that keeping x1 clamped at the floor enables x3 saturation; any drift in x4 must stay coupled to high x2/x3.
# """
# )


