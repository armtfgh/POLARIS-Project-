# Knowledge Extraction Pipeline Improvements

## Overview

The knowledge extraction pipeline in `cartographer.py` has been significantly improved to address critical issues that were causing unstable and suboptimal NDCG scores (ranging from 0.28-0.51).

## Problems Identified

### 1. **Token Truncation (Critical)**
**Issue:** Descriptions were being cut off mid-sentence due to `max_tokens=600`

**Evidence from `knowledge_df.csv`:**
- Row 1 ends with: "...the reaction is primarily driven by x2 (Ald"
- Row 3 ends incomplete

**Impact:** Critical information lost before scoring step

**Fix:** Increased to `max_tokens=1500` for direct JSON, `1200` for legacy

---

### 2. **Two-Step LLM Translation Loss**
**Issue:** Lossy two-stage process:
1. Stage 1: GPT-4o generates natural language prose
2. Stage 2: Another LLM converts prose → JSON

Each translation introduces errors and information loss.

**Impact:** Analyst creates free-form prose, but scorer expects structured JSON. Translation is unreliable.

**Fix:** New `synthesize_landscape_readout_direct()` generates JSON in **one step**

---

### 3. **Ground Truth Contamination**
**Issue:** `build_scoring_prompt_from_description()` includes `UGI_DATASET_CONTEXT` with ground truth hints:

```python
x1 = amine_mM → excessive amine suppresses yield.
x2 = aldehyde_mM → moderately positive lever...
x3 = isocyanide_mM → strongest positive driver; peak near 280–300 mM.
```

**Impact:** LLM told ground truth while trying to learn from observations → confusion about what to infer vs. what it "knows"

**Fix:** New prompts contain **zero** external chemical knowledge - only observed data

---

### 4. **Inadequate Data Representation**
**Issue:** Old `_format_observations_for_analyst()` only showed:
- Top-k and bottom-k samples (max 30 total)
- Pearson correlations (assumes linearity)
- No spatial distribution info

**Impact:** For 4D space with ~16-26 samples (early iterations), showing only extremes misses:
- Non-monotonic relationships
- Interaction effects in middle range
- Spatial clustering/gaps

**Fix:** New `_format_observations_for_analyst_enhanced()` adds:
- Binned analysis (split at median, compare mean yields)
- Extended statistics (Q1, Q3, std)
- Warnings for small sample sizes
- Clear normalization reminders

---

### 5. **Sparse Data Challenges**
**Issue:** At iteration 19 (first checkpoint): 26 samples in 4D space
- Only ~6.5 samples per dimension
- Correlations unstable with small n
- LLM struggles to distinguish signal from noise

**Fix:** New prompt explicitly:
- Instructs LLM to use LOW confidence when n < 50
- Recommends simpler models (fewer interactions, broader bumps) for sparse data
- Emphasizes conservative estimation

---

### 6. **Inconsistent Description Quality**
**Issue:** Analyst outputs varied wildly (from `knowledge_df.csv`):
- Some focus on correlations: "corr = 0.4594"
- Some on regions: "peak around 0.3 equivalents"
- Some emphasize interactions, others don't

**Impact:** JSON conversion step becomes unreliable with inconsistent inputs

**Fix:** New prompt provides **strict structured format** with:
- Clear field specifications
- Valid value ranges
- Explicit examples
- Required fields enforced

---

## New Features

### 1. Direct JSON Generation (`ANALYST_SYSTEM_PROMPT_DIRECT_JSON`)

**What it does:**
- Single-step: observed data → JSON readout (no intermediate prose)
- Eliminates translation losses
- Enforces structured output with `response_format={"type": "json_object"}`

**JSON Format:**
```json
{
  "effects": {
    "x1": {"effect": "increasing|decreasing|nonmonotone-peak|nonmonotone-valley|flat",
           "scale": 0.0-1.0, "confidence": 0.0-1.0, "range_hint": [low, high]},
    ...
  },
  "interactions": [
    {"vars": ["xi", "xj"], "type": "synergy|tradeoff",
     "scale": 0.0-1.0, "confidence": 0.0-1.0, "note": "explanation"}
  ],
  "bumps": [
    {"mu": [x1, x2, x3, x4], "sigma": [s1, s2, s3, s4], "amp": 0.05-0.15}
  ]
}
```

---

### 2. Enhanced Data Formatting

**Binned Analysis Example:**
```
=== BINNED ANALYSIS (split at median) ===
x1: low half mean_yield=0.0823, high half mean_yield=0.0651, diff=-0.0172
x2: low half mean_yield=0.0612, high half mean_yield=0.0862, diff=+0.0250
x3: low half mean_yield=0.0545, high half mean_yield=0.0929, diff=+0.0384
x4: low half mean_yield=0.0298, high half mean_yield=0.1176, diff=+0.0878
```

**Benefit:** LLM can see directional trends clearly without assuming linearity

---

### 3. Zero Ground Truth Contamination

**Before:**
```python
# OLD: Included in scoring prompt
UGI_DATASET_CONTEXT = """
x1 = amine_mM → excessive amine suppresses yield.
x2 = aldehyde_mM → moderately positive lever...
"""
```

**After:**
```python
# NEW: Only observed data
"Base ALL conclusions ONLY on observed data - ignore external chemical knowledge"
```

---

## Usage

### Quick Start (Improved Pipeline)

```python
from cartographer import run_cartographer_analyst_pipeline_improved, plot_knowledge_curve

hist_df, knowledge_df = run_cartographer_analyst_pipeline_improved(
    csv_path="ugi_merged_dataset.csv",
    budget=100,
    n_init=6,
    seed=42,
    analyst_every=20,
    analyst_model="gpt-4o",
    api_key=None,  # or your OpenAI API key
    use_direct_json=True,  # NEW: Direct JSON (recommended)
)

print(knowledge_df[["iter", "n_obs", "ndcg"]])
plot_knowledge_curve(knowledge_df)
```

### Running Comparison Test

```bash
# Full comparison (takes ~5-10 minutes)
python test_improved_cartographer.py

# Quick test (2 checkpoints)
python test_improved_cartographer.py --quick
```

---

## Expected Improvements

Based on the fixes, you should see:

1. **Higher NDCG scores** (0.45-0.55 range instead of 0.28-0.51)
2. **More stable scores** (lower standard deviation across checkpoints)
3. **No truncation** (complete descriptions/readouts)
4. **Better early performance** (stronger signal from sparse data)
5. **More interpretable outputs** (structured JSON readouts)

---

## Recommended Next Steps

### Immediate (Already Implemented)
- [x] Increase max_tokens to 1500
- [x] Remove ground truth contamination
- [x] Implement direct JSON generation
- [x] Add binned analysis to data representation

### Future Enhancements
1. **Increase initial samples:** 12-15 instead of 6 for more stable early estimates
2. **Check less frequently:** Every 20-30 iterations (currently 10) to allow more data accumulation
3. **Add uncertainty quantification:** Report confidence intervals on NDCG scores
4. **Visualize readouts:** Plot learned priors vs ground truth landscape
5. **Ensemble multiple analysts:** Average over 3-5 independent LLM calls to reduce variance
6. **Active learning for analyst:** Train analyst to focus on informative regions

---

## Backward Compatibility

The original pipeline is preserved:
- `run_cartographer_analyst_pipeline()` - legacy two-step approach
- `synthesize_landscape_description()` - legacy prose generation
- Old behavior is unchanged for existing code

---

## Key Functions

| Function | Purpose | Status |
|----------|---------|--------|
| `synthesize_landscape_readout_direct()` | **NEW**: Direct JSON generation | ✓ Recommended |
| `_format_observations_for_analyst_enhanced()` | **NEW**: Binned analysis + warnings | ✓ Recommended |
| `run_cartographer_analyst_pipeline_improved()` | **NEW**: Complete improved pipeline | ✓ Recommended |
| `synthesize_landscape_description()` | LEGACY: Prose generation | Still available |
| `run_cartographer_analyst_pipeline()` | LEGACY: Two-step pipeline | Still available |

---

## Technical Details

### Prompt Engineering Improvements

1. **Explicit field constraints:**
   - `effect: MUST be one of {"increasing", "decreasing", ...}`
   - `scale: range [0.0, 1.0]`
   - `confidence: range [0.0, 1.0]`

2. **Normalization clarity:**
   - "range_hint must be in NORMALIZED [0,1] space (not raw units!)"
   - Explicit formula: `(value - min) / (max - min)`

3. **Uncertainty guidance:**
   - "Use LOW confidence (<0.5) when n_samples < 50"
   - "Be conservative: underestimating is safer than hallucinating"

4. **Sparse data handling:**
   - "With sparse data (<30 samples), prefer simple models"
   - Fewer interactions, broader bumps

### Error Handling

```python
try:
    readout = json.loads(llm_response)
except json.JSONDecodeError as e:
    # Fallback: return flat prior
    readout = {"effects": {...}, "interactions": [], "bumps": []}
```

Ensures pipeline never crashes on malformed LLM output.

---

## Questions?

For issues or questions about the improvements, check:
1. `test_improved_cartographer.py` - working examples
2. `cartographer.py` - full implementation
3. `knowledge_df.csv` - original problematic outputs

---

**Last Updated:** 2025-12-18
**Author:** Claude Code (Sonnet 4.5)
