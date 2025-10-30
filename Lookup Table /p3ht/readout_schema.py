# readout_schema.py
from typing import Dict, Any
from prior_gp import Prior

# Standardized readout:
# {
#   "effects": {
#       "x1": {"effect": "increase|decrease|nonmonotone-peak|nonmonotone-valley|flat",
#              "scale": float, "confidence": float, "range_hint": [low, high] (optional)},
#       "x2": {...}
#   },
#   "interactions": [{"pair":["x1","x2"], "type":"synergy|antagonism", "confidence": float}],
#   "bumps": [{"mu":[x1,x2], "sigma": float, "amp": float}]
# }

def flat_readout() -> Dict[str, Any]:
    return {
        "effects": {
            "x1": {"effect": "flat", "scale": 0.0, "confidence": 0.0},
            "x2": {"effect": "flat", "scale": 0.0, "confidence": 0.0},
        },
        "interactions": [],
        "bumps": [],
    }

def readout_to_prior(ro: Dict[str, Any]) -> Prior:
    effects = ro.get("effects", {})
    inter   = ro.get("interactions", [])
    bumps   = ro.get("bumps", [])
    return Prior(effects=effects, interactions=inter, bumps=bumps)
