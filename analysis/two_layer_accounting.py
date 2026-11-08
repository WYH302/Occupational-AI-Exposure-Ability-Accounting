"""Exact FRS-style ability accounting and non-causal profile diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def _ability_key(values: pd.Series) -> pd.Series:
    return (
        values.astype(str)
        .str.lower()
        .str.replace(" ", "", regex=False)
        .str.replace("-", "", regex=False)
    )


def _validate_probability_vector(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all() or (array < 0).any():
        raise ValueError(f"{name} must be a finite nonnegative vector")
    total = float(array.sum())
    if total <= 0:
        raise ValueError(f"{name} must have positive mass")
    return array / total


def jensen_shannon_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return base-2 Jensen--Shannon distance on two nonnegative vectors."""
    p = _validate_probability_vector(left, "left")
    q = _validate_probability_vector(right, "right")
    midpoint = 0.5 * (p + q)

    def divergence(x: np.ndarray) -> float:
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / midpoint[mask])))

    distance_squared = max(0.0, 0.5 * divergence(p) + 0.5 * divergence(q))
    return float(np.sqrt(distance_squared))


def compute_profile_diagnostics(
    ledger: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add positive-profile shares and occupation-level profile diagnostics."""
    required = {"soc", "Ability", "positive_contribution", "reuse_weight"}
    missing = required.difference(ledger.columns)
    if missing:
        raise ValueError(f"profile ledger is missing columns: {sorted(missing)}")
    detail = ledger.copy()
    if "negative_contribution" not in detail.columns:
        detail["negative_contribution"] = 0.0
    positive_mass = detail.groupby("soc")["positive_contribution"].transform("sum")
    detail["positive_profile_share"] = np.where(
        positive_mass > 0,
        detail["positive_contribution"] / positive_mass,
        np.nan,
    )
    total_mass = detail.groupby("soc")[
        ["positive_contribution", "negative_contribution"]
    ].transform("sum").sum(axis=1)
    detail["positive_signed_share"] = np.where(
        total_mass > 0,
        detail["positive_contribution"] / total_mass,
        np.nan,
    )
    detail["negative_signed_share"] = np.where(
        total_mass > 0,
        detail["negative_contribution"] / total_mass,
        np.nan,
    )

    rows: list[dict[str, float | str]] = []
    for soc, group in detail.groupby("soc", sort=True):
        shares = group["positive_profile_share"].to_numpy(dtype=float)
        if np.isnan(shares).all():
            positive_reuse = np.nan
            concentration = np.nan
            effective_count = np.nan
            support_count = 0
        else:
            support_count = int(np.sum(shares > 0))
            positive_reuse = float(
                np.nansum(shares * group["reuse_weight"].to_numpy(dtype=float))
            )
            concentration = float(np.nansum(np.square(shares)))
            effective_count = 1.0 / concentration if concentration > 0 else np.nan
        rows.append(
            {
                "soc": soc,
                "positive_reuse": positive_reuse,
                "exposure_concentration": concentration,
                "effective_ability_count": effective_count,
                "positive_support_count": support_count,
            }
        )
    return detail, pd.DataFrame(rows)


def build_exact_accounting(
    weights_and_scores: pd.DataFrame,
    reference_center: float | None = None,
    reference_scale: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct an identity-preserving standardized ability ledger.

    The input must have one row per occupation and ability. Weights are
    normalized within occupation. The resulting ability contributions sum to
    the standardized occupation score to floating-point precision.
    """
    required = {"soc", "Ability", "effective_weight", "ability_score"}
    missing = required.difference(weights_and_scores.columns)
    if missing:
        raise ValueError(f"accounting input is missing columns: {sorted(missing)}")
    detail = weights_and_scores.copy()
    if detail.duplicated(["soc", "Ability"]).any():
        raise ValueError("accounting input must have one row per occupation and ability")
    numeric = detail[["effective_weight", "ability_score"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (detail["effective_weight"] < 0).any():
        raise ValueError("weights and ability scores must be finite; weights must be nonnegative")
    detail["effective_weight"] = detail["effective_weight"].astype("float64")
    detail["ability_score"] = detail["ability_score"].astype("float64")
    weight_sum = detail.groupby("soc")["effective_weight"].transform("sum")
    if (weight_sum <= 0).any():
        raise ValueError("every occupation must have positive total weight")
    detail["effective_weight"] = detail["effective_weight"] / weight_sum
    detail["raw_contribution"] = detail["effective_weight"] * detail["ability_score"]
    raw_scores = detail.groupby("soc", as_index=False)["raw_contribution"].sum()
    raw_scores = raw_scores.rename(columns={"raw_contribution": "raw_score"})
    center = (
        float(raw_scores["raw_score"].mean())
        if reference_center is None
        else float(reference_center)
    )
    scale = (
        float(raw_scores["raw_score"].std(ddof=1))
        if reference_scale is None
        else float(reference_scale)
    )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("occupation scores must have positive cross-occupation variation")

    detail["standardized_contribution"] = (
        detail["effective_weight"] * (detail["ability_score"] - center) / scale
    )
    detail["positive_contribution"] = detail["standardized_contribution"].clip(lower=0)
    detail["negative_contribution"] = (-detail["standardized_contribution"]).clip(lower=0)
    ability_totals = detail.groupby("Ability")["effective_weight"].transform("sum")
    detail["reuse_weight"] = ability_totals / float(
        detail.groupby("Ability")["effective_weight"].sum().mean()
    )
    detail, profiles = compute_profile_diagnostics(detail)

    occupations = raw_scores.copy()
    occupations["standardized_score"] = (occupations["raw_score"] - center) / scale
    accounts = (
        detail.groupby("soc", as_index=False)
        .agg(
            positive_account=("positive_contribution", "sum"),
            negative_account=("negative_contribution", "sum"),
            ledger_sum=("standardized_contribution", "sum"),
        )
    )
    occupations = occupations.merge(accounts, on="soc", validate="one_to_one")
    occupations = occupations.merge(profiles, on="soc", validate="one_to_one")
    occupations["reconstruction_error"] = (
        occupations["ledger_sum"] - occupations["standardized_score"]
    )
    detail["score_center"] = center
    detail["score_scale"] = scale
    return detail.sort_values(["soc", "Ability"]).reset_index(drop=True), occupations.sort_values(
        "soc"
    ).reset_index(drop=True)


def _frs_ability_scores(
    applications: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    merged = applications.merge(mapping, on="applications", how="inner", validate="one_to_one")
    ability_columns = [
        column for column in mapping.columns if column not in {"application_id", "applications"}
    ]
    weighted = merged[ability_columns].multiply(merged["binary_all"], axis=0)
    scores = weighted.sum(axis=0).rename("ability_score").reset_index()
    return scores.rename(columns={"index": "ability_key"})


def _frs_effective_weights(abilities: pd.DataFrame) -> pd.DataFrame:
    data = abilities.rename(
        columns={
            "onetsoccode": "onet_soc",
            "elementid": "element_id",
            "elementname": "Ability",
            "scaleid": "scale",
            "datavalue": "value",
        }
    ).copy()
    data["ability_key"] = _ability_key(data["Ability"])
    data.loc[
        data["ability_key"].eq("visualcolordetermination"), "ability_key"
    ] = "visualcolordiscrimination"
    data.loc[
        data["ability_key"].eq("visualcolordiscrimination"), "Ability"
    ] = "Visual Color Discrimination"
    wide = (
        data[data["scale"].isin(["IM", "LV"])]
        .pivot_table(
            index=["onet_soc", "element_id", "Ability", "ability_key"],
            columns="scale",
            values="value",
            aggfunc="max",
        )
        .reset_index()
    )
    if wide[["IM", "LV"]].isna().any().any():
        raise ValueError("FRS replication weights require both importance and level ratings")
    wide["scalar"] = (wide["IM"] / 5.0) * (wide["LV"] / 7.0)
    wide["soc"] = wide["onet_soc"].astype(str).str.slice(0, 7)
    effective = (
        wide.groupby(["soc", "Ability", "ability_key"], as_index=False)["scalar"]
        .sum()
        .rename(columns={"scalar": "effective_weight"})
    )
    return effective


def build_frs_replication_accounting(
    raw_dir: Path = RAW_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebuild the archived FRS score and its exact 52-ability ledger."""
    abilities = pd.read_stata(raw_dir / "frs_abilities_2020.dta")
    applications = pd.read_stata(raw_dir / "frs_application_list.dta")
    mapping = pd.read_stata(raw_dir / "frs_mturk_mapping_matrix.dta")
    published = pd.read_excel(raw_dir / "AIOE_DataAppendix.xlsx", sheet_name="Appendix A")
    published = published.rename(
        columns={
            "SOC Code": "soc",
            "Occupation Title": "occupation_title",
            "AIOE": "published_aioe",
        }
    )

    weights = _frs_effective_weights(abilities)
    scores = _frs_ability_scores(applications, mapping)
    accounting_input = weights.merge(scores, on="ability_key", how="left", validate="many_to_one")
    if accounting_input["ability_score"].isna().any():
        missing = sorted(accounting_input.loc[accounting_input["ability_score"].isna(), "Ability"].unique())
        raise ValueError(f"FRS ability scores missing for: {missing}")
    ledger, occupations = build_exact_accounting(accounting_input)
    occupations = occupations.merge(
        published[["soc", "occupation_title", "published_aioe"]],
        on="soc",
        how="left",
        validate="one_to_one",
    )
    published_matches = int(occupations["published_aioe"].notna().sum())
    if occupations.shape[0] != 774 or published_matches != 774:
        raise ValueError(
            "expected 774/774 published-score matches; "
            f"found {published_matches}/{occupations.shape[0]}"
        )
    occupations["published_residual"] = (
        occupations["published_aioe"] - occupations["standardized_score"]
    )
    return ledger, occupations


def build_nearest_score_pairs(
    occupations: pd.DataFrame,
    score_col: str,
    tolerance: float | None = None,
) -> pd.DataFrame:
    """Match each occupation to its nearest score neighbor and deduplicate pairs."""
    required = {"soc", score_col}
    missing = required.difference(occupations.columns)
    if missing:
        raise ValueError(f"matching input is missing columns: {sorted(missing)}")
    frame = occupations[["soc", score_col]].dropna().copy()
    frame["soc"] = frame["soc"].astype(str)
    frame = frame.sort_values("soc").reset_index(drop=True)
    rows: list[dict[str, float | str]] = []
    for _, row in frame.iterrows():
        candidates = frame[frame["soc"] != row["soc"]].copy()
        candidates["score_gap"] = (candidates[score_col] - row[score_col]).abs()
        candidates = candidates.sort_values(["score_gap", "soc"])
        if candidates.empty:
            continue
        neighbor = candidates.iloc[0]
        gap = float(neighbor["score_gap"])
        if tolerance is not None and gap > tolerance:
            continue
        left, right = sorted([str(row["soc"]), str(neighbor["soc"])])
        rows.append({"soc_left": left, "soc_right": right, "score_gap": gap})
    if not rows:
        return pd.DataFrame(columns=["soc_left", "soc_right", "score_gap"])
    return (
        pd.DataFrame(rows)
        .sort_values(["soc_left", "soc_right", "score_gap"])
        .drop_duplicates(["soc_left", "soc_right"], keep="first")
        .reset_index(drop=True)
    )


def _onet_abilities_path(release: str, raw_dir: Path) -> Path:
    directory = f"db_{release}_excel"
    candidates = [
        raw_dir / directory / "Abilities.xlsx",
        raw_dir / directory / directory / "Abilities.xlsx",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"O*NET {release.replace('_', '.')} Abilities.xlsx was not found")


def _onet_effective_weights(
    release: str,
    raw_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = pd.read_excel(_onet_abilities_path(release, raw_dir))
    data = data[data["Scale ID"].isin(["IM", "LV"])].copy()
    data = data.rename(
        columns={
            "O*NET-SOC Code": "onet_soc",
            "Element ID": "element_id",
            "Element Name": "Ability",
            "Scale ID": "scale",
            "Data Value": "value",
            "Title": "occupation_title",
        }
    )
    data["ability_key"] = _ability_key(data["Ability"]).replace(
        {"visualcolordetermination": "visualcolordiscrimination"}
    )
    data.loc[
        data["ability_key"].eq("visualcolordiscrimination"), "Ability"
    ] = "Visual Color Discrimination"
    wide = (
        data.pivot_table(
            index=["onet_soc", "element_id", "Ability", "ability_key"],
            columns="scale",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )
    if wide[["IM", "LV"]].isna().any().any():
        raise ValueError(f"O*NET {release} profiles require both importance and level ratings")
    wide["scalar"] = (wide["IM"].astype(float) / 5.0) * (wide["LV"].astype(float) / 7.0)
    wide["soc"] = wide["onet_soc"].astype(str).str.slice(0, 7)
    effective = (
        wide.groupby(["soc", "Ability", "ability_key"], as_index=False)["scalar"]
        .sum()
        .rename(columns={"scalar": "effective_weight"})
    )
    titles = data.assign(soc=data["onet_soc"].astype(str).str.slice(0, 7))
    titles = titles.groupby("soc", as_index=False)["occupation_title"].first()
    return effective, titles


def build_onet_release_accounting(
    release: str,
    raw_dir: Path = RAW_DIR,
    reference_center: float | None = None,
    reference_scale: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build an exact FRS-style ledger using a declared O*NET release."""
    applications = pd.read_stata(raw_dir / "frs_application_list.dta")
    mapping = pd.read_stata(raw_dir / "frs_mturk_mapping_matrix.dta")
    published = pd.read_excel(raw_dir / "AIOE_DataAppendix.xlsx", sheet_name="Appendix A")
    published = published.rename(columns={"SOC Code": "soc", "AIOE": "published_aioe"})
    weights, titles = _onet_effective_weights(release, raw_dir)
    scores = _frs_ability_scores(applications, mapping)
    accounting_input = weights.merge(scores, on="ability_key", how="left", validate="many_to_one")
    if accounting_input["ability_score"].isna().any():
        missing = sorted(accounting_input.loc[accounting_input["ability_score"].isna(), "Ability"].unique())
        raise ValueError(f"FRS ability scores missing for: {missing}")
    ledger, occupations = build_exact_accounting(
        accounting_input,
        reference_center=reference_center,
        reference_scale=reference_scale,
    )
    ledger.insert(0, "O*NET release", release.replace("_", "."))
    occupations.insert(0, "O*NET release", release.replace("_", "."))
    occupations = occupations.merge(titles, on="soc", how="left", validate="one_to_one")
    occupations = occupations.merge(
        published[["soc", "published_aioe"]], on="soc", how="left", validate="one_to_one"
    )
    return ledger, occupations


def _profile_matrix(ledger: pd.DataFrame) -> pd.DataFrame:
    required = {"soc", "Ability", "positive_profile_share"}
    missing = required.difference(ledger.columns)
    if missing:
        raise ValueError(f"profile ledger is missing columns: {sorted(missing)}")
    return ledger.pivot(index="soc", columns="Ability", values="positive_profile_share").fillna(0.0)


def _contribution_mass_profile_matrix(ledger: pd.DataFrame) -> pd.DataFrame:
    """Return a compact 52-ability contribution-mass composition.

    Each ability's contribution sign is fixed by its application score and the
    declared standardization origin. Ability identity therefore retains the
    sign partition, and duplicating every ability into positive/negative cells
    would add zero information.
    """
    required = {"soc", "Ability", "standardized_contribution"}
    missing = required.difference(ledger.columns)
    if missing:
        raise ValueError(f"contribution profile ledger is missing columns: {sorted(missing)}")
    detail = ledger[["soc", "Ability", "standardized_contribution"]].copy()
    nonzero = detail[detail["standardized_contribution"].ne(0)].copy()
    nonzero["sign"] = np.sign(nonzero["standardized_contribution"])
    if nonzero.groupby("Ability")["sign"].nunique().gt(1).any():
        raise ValueError("ability contribution signs must be fixed under a declared origin")
    detail["contribution_mass"] = detail["standardized_contribution"].abs()
    total = detail.groupby("soc")["contribution_mass"].transform("sum")
    if (total <= 0).any():
        raise ValueError("every contribution profile must have positive absolute mass")
    detail["contribution_mass_share"] = detail["contribution_mass"] / total
    return detail.pivot(
        index="soc", columns="Ability", values="contribution_mass_share"
    ).fillna(0.0).sort_index(axis=1)


def _distance_metrics(left: np.ndarray, right: np.ndarray) -> tuple[float, float, float]:
    left = _validate_probability_vector(left, "left")
    right = _validate_probability_vector(right, "right")
    jsd = jensen_shannon_distance(left, right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    cosine = 1.0 - float(np.dot(left, right)) / denominator if denominator > 0 else np.nan
    cosine = float(np.clip(cosine, 0.0, 1.0)) if np.isfinite(cosine) else np.nan
    manhattan = float(np.sum(np.abs(left - right)))
    return jsd, cosine, manhattan


def build_score_band_pairs(
    occupations: pd.DataFrame,
    score_col: str,
    tolerance: float,
    require_different_major: bool = True,
) -> pd.DataFrame:
    """Enumerate every unique occupation pair inside a declared score band."""
    if tolerance < 0:
        raise ValueError("tolerance must be nonnegative")
    required = {"soc", score_col}
    missing = required.difference(occupations.columns)
    if missing:
        raise ValueError(f"matching input is missing columns: {sorted(missing)}")
    frame = occupations[["soc", score_col]].dropna().copy()
    frame["soc"] = frame["soc"].astype(str)
    frame = frame.sort_values("soc").reset_index(drop=True)
    rows: list[dict[str, float | str]] = []
    for left_index in range(frame.shape[0]):
        left = frame.iloc[left_index]
        for right_index in range(left_index + 1, frame.shape[0]):
            right = frame.iloc[right_index]
            if require_different_major and left["soc"][:2] == right["soc"][:2]:
                continue
            gap = abs(float(left[score_col]) - float(right[score_col]))
            if gap <= tolerance:
                rows.append(
                    {
                        "soc_left": left["soc"],
                        "soc_right": right["soc"],
                        "score_gap": gap,
                    }
                )
    return pd.DataFrame(rows, columns=["soc_left", "soc_right", "score_gap"])


def build_profile_distance_table(
    ledger: pd.DataFrame,
    occupations: pd.DataFrame,
    score_col: str = "published_aioe",
    secondary_score_col: str | None = None,
    tolerance: float | None = 0.025,
    require_different_major: bool = True,
) -> pd.DataFrame:
    """Measure contribution-profile divergence inside one or two score bands."""
    if tolerance is None:
        raise ValueError("profile distance matching requires a finite tolerance")
    pairs = build_score_band_pairs(
        occupations,
        score_col=score_col,
        tolerance=tolerance,
        require_different_major=require_different_major,
    )
    if pairs.empty:
        return pairs.assign(
            jensen_shannon_distance=pd.Series(dtype=float),
            cosine_distance=pd.Series(dtype=float),
            manhattan_distance=pd.Series(dtype=float),
        )
    if secondary_score_col is not None and secondary_score_col not in occupations.columns:
        raise ValueError(f"matching input is missing column: {secondary_score_col}")
    profiles = _contribution_mass_profile_matrix(ledger)
    rows: list[dict[str, float | str]] = []
    metadata = occupations.set_index("soc")
    for row in pairs.itertuples(index=False):
        if row.soc_left not in profiles.index or row.soc_right not in profiles.index:
            continue
        secondary_gap = np.nan
        if secondary_score_col is not None:
            secondary_gap = abs(
                float(metadata.at[row.soc_left, secondary_score_col])
                - float(metadata.at[row.soc_right, secondary_score_col])
            )
            if not np.isfinite(secondary_gap) or secondary_gap > tolerance:
                continue
        jsd, cosine, manhattan = _distance_metrics(
            profiles.loc[row.soc_left].to_numpy(dtype=float),
            profiles.loc[row.soc_right].to_numpy(dtype=float),
        )
        rows.append(
            {
                "soc_left": row.soc_left,
                "occupation_left": metadata.at[row.soc_left, "occupation_title"],
                "soc_right": row.soc_right,
                "occupation_right": metadata.at[row.soc_right, "occupation_title"],
                "score_left": float(metadata.at[row.soc_left, score_col]),
                "score_right": float(metadata.at[row.soc_right, score_col]),
                "score_gap": float(row.score_gap),
                "secondary_score_gap": secondary_gap,
                "jensen_shannon_distance": jsd,
                "cosine_distance": cosine,
                "manhattan_distance": manhattan,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "soc_left",
                "occupation_left",
                "soc_right",
                "occupation_right",
                "score_left",
                "score_right",
                "score_gap",
                "secondary_score_gap",
                "jensen_shannon_distance",
                "cosine_distance",
                "manhattan_distance",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["score_gap", "soc_left", "soc_right"]
    ).reset_index(drop=True)


def build_version_drift_table(
    earlier_ledger: pd.DataFrame,
    later_ledger: pd.DataFrame,
    earlier_label: str,
    later_label: str,
    weight_update_tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Measure profile drift and identify occupations with revised ability weights."""
    if weight_update_tolerance < 0:
        raise ValueError("weight_update_tolerance must be nonnegative")
    required = {"soc", "Ability", "effective_weight"}
    for label, ledger in (("earlier", earlier_ledger), ("later", later_ledger)):
        missing = required.difference(ledger.columns)
        if missing:
            raise ValueError(
                f"{label} version ledger is missing columns: {sorted(missing)}"
            )
    earlier = _contribution_mass_profile_matrix(earlier_ledger)
    later = _contribution_mass_profile_matrix(later_ledger)
    earlier_weights = earlier_ledger.pivot(
        index="soc", columns="Ability", values="effective_weight"
    )
    later_weights = later_ledger.pivot(
        index="soc", columns="Ability", values="effective_weight"
    )
    common_soc = sorted(set(earlier.index).intersection(later.index))
    common_abilities = sorted(set(earlier.columns).intersection(later.columns))
    if len(common_abilities) != 52:
        raise ValueError(
            "expected exactly 52 common abilities for version drift; "
            f"found {len(common_abilities)}"
        )
    rows: list[dict[str, float | str]] = []
    for soc in common_soc:
        weight_change = np.abs(
            earlier_weights.loc[soc, common_abilities].to_numpy(dtype=float)
            - later_weights.loc[soc, common_abilities].to_numpy(dtype=float)
        )
        updated_ability_count = int(
            np.sum(weight_change > weight_update_tolerance)
        )
        jsd, cosine, manhattan = _distance_metrics(
            earlier.loc[soc, common_abilities].to_numpy(dtype=float),
            later.loc[soc, common_abilities].to_numpy(dtype=float),
        )
        rows.append(
            {
                "soc": soc,
                "earlier_release": earlier_label,
                "later_release": later_label,
                "jensen_shannon_distance": jsd,
                "cosine_distance": cosine,
                "manhattan_distance": manhattan,
                "updated_ability_count": updated_ability_count,
                "any_ability_weight_updated": updated_ability_count > 0,
                "maximum_absolute_weight_change": float(np.max(weight_change)),
                "weight_update_tolerance": float(weight_update_tolerance),
            }
        )
    return pd.DataFrame(rows)
