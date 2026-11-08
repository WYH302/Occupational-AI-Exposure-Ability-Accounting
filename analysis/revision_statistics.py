from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


ABILITY_NAME_ALIASES = {
    "Visual Color Discrimination": "Visual Color Determination",
}


def deflate_log_values(
    log_values: pd.Series,
    years: pd.Series,
    annual_price_index: dict[int, float],
    *,
    base_year: int,
) -> pd.Series:
    """Convert logged nominal values to a declared base year's prices."""
    if base_year not in annual_price_index:
        raise KeyError(f"Base year {base_year} is absent from the price index")
    mapped = years.map(annual_price_index)
    if mapped.isna().any():
        missing = sorted(years[mapped.isna()].astype(int).unique().tolist())
        raise KeyError(f"Price index missing years: {missing}")
    if (mapped <= 0).any() or annual_price_index[base_year] <= 0:
        raise ValueError("Price-index values must be positive")
    adjustment = np.log(float(annual_price_index[base_year]) / mapped.astype(float))
    return log_values.astype(float) + adjustment


def normalize_ability_names(
    frame: pd.DataFrame,
    column: str = "Element Name",
) -> pd.DataFrame:
    """Return a copy with documented O*NET/FRS ability-name aliases resolved."""
    normalized = frame.copy()
    normalized[column] = normalized[column].replace(ABILITY_NAME_ALIASES)
    return normalized


def robust_joint_pvalue(result: object, parameter_names: Sequence[str]) -> float:
    """Test named coefficients jointly using the covariance attached to ``result``."""
    exog_names = list(result.model.exog_names)
    missing = [name for name in parameter_names if name not in exog_names]
    if missing:
        raise KeyError(f"Parameters absent from fitted model: {missing}")

    restrictions = np.zeros((len(parameter_names), len(exog_names)), dtype=float)
    for row, name in enumerate(parameter_names):
        restrictions[row, exog_names.index(name)] = 1.0
    return float(result.wald_test(restrictions, scalar=True).pvalue)


def benjamini_hochberg(pvalues: Iterable[float]) -> np.ndarray:
    """Return Benjamini--Hochberg adjusted p-values in their original order."""
    values = np.asarray(list(pvalues), dtype=float)
    if values.ndim != 1:
        raise ValueError("pvalues must be one-dimensional")
    if values.size == 0:
        return values
    if np.any(~np.isfinite(values)) or np.any((values < 0) | (values > 1)):
        raise ValueError("pvalues must be finite values in [0, 1]")

    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = ranked * values.size / np.arange(1, values.size + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def aggregate_exposure_components(
    occupation_ability: pd.DataFrame,
    *,
    hub_col: str,
    shift: float = 0.0,
    threshold: float = 0.0,
    exposure_col: str = "ai_exposure",
) -> pd.DataFrame:
    """Aggregate positive, relative-low, and reuse loads after a declared transform."""
    required = {"soc", "Element Name", "w_norm", hub_col, exposure_col}
    missing = sorted(required.difference(occupation_ability.columns))
    if missing:
        raise KeyError(f"Missing occupation-ability columns: {missing}")

    temp = occupation_ability.copy()
    centered = temp[exposure_col].astype(float) + float(shift) - float(threshold)
    weighted_hub = temp["w_norm"].astype(float) * temp[hub_col].astype(float)
    temp["substitution_pressure"] = weighted_hub * centered.clip(lower=0.0)
    temp["relative_low_exposure_load"] = weighted_hub * (-centered).clip(lower=0.0)
    temp["reuse_intensity"] = weighted_hub

    return (
        temp.groupby("soc", as_index=False)
        .agg(
            substitution_pressure=("substitution_pressure", "sum"),
            relative_low_exposure_load=("relative_low_exposure_load", "sum"),
            reuse_intensity=("reuse_intensity", "sum"),
            ability_count=("Element Name", "nunique"),
        )
    )
