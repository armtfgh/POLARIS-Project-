# readout_schema.py
from __future__ import annotations

from typing import Dict, Any, Iterable, List, Optional

from prior_gp import Prior


def _feature_keys(
    feature_names: Optional[Iterable[str]] = None,
    *,
    n_fallback: int = 4,
) -> List[str]:
    if feature_names is not None:
        names = [str(name) for name in feature_names]
        if names:
            return names
    return [f"x{i+1}" for i in range(n_fallback)]


def flat_readout(
    feature_names: Optional[Iterable[str]] = None,
    *,
    n_fallback: int = 4,
) -> Dict[str, Any]:
    names = _feature_keys(feature_names, n_fallback=n_fallback)
    effects = {
        name: {"effect": "flat", "scale": 0.0, "confidence": 0.0}
        for name in names
    }
    return {
        "effects": effects,
        "interactions": [],
        "bumps": [],
    }


def readout_to_prior(
    ro: Dict[str, Any],
    *,
    feature_names: Optional[Iterable[str]] = None,
) -> Prior:
    effects = ro.get("effects", {})
    inter = ro.get("interactions", [])
    bumps = ro.get("bumps", [])
    names = _feature_keys(feature_names) if feature_names is not None else None
    return Prior(
        effects=effects,
        interactions=inter,
        bumps=bumps,
        feature_names=names,
    )

