"""Build the two-layer accounting, profile, and boundary-check manuscript assets."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib.patches import FancyBboxPatch
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from analysis.build_manuscript_assets import build_measure_frame
from analysis.build_validation_assets import load_census_soc_crosswalk
from analysis.revision_experiments import (
    CPI_U_ANNUAL,
    build_acs_change,
    build_oe_change,
    coefficient,
    deflate_log_values,
)
from analysis.revision_statistics import robust_joint_pvalue
from analysis.two_layer_accounting import (
    build_frs_replication_accounting,
    build_onet_release_accounting,
    build_profile_distance_table,
    build_version_drift_table,
)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "mathtext.fontset": "stix",
        "text.color": "#000000",
        "axes.edgecolor": "#000000",
        "axes.labelcolor": "#000000",
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    }
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
PAPER_TABLES = ROOT / "paper" / "tables"
PAPER_FIGURES = ROOT / "paper" / "figures"
ACTIVE_PAPER_TABLES = ROOT / "paper" / "reconstructed_latest" / "tables"
MASTER = RESULTS / "master"
RNG_SEED = 20260717
PROFILE_TERMS = ["positive_reuse_std", "effective_ability_count_std"]
ACTIVE_TABLE_STEMS = {
    "table_exact_accounting_audit",
    "table_profile_summary_main",
    "table_profile_distance_benchmark_main",
    "table_profile_external_association_main",
}
PROFILE_LABELS = {
    "positive_reuse_std": "Positive-profile reuse",
    "effective_ability_count_std": "Effective ability count",
}


def ensure_directories() -> None:
    for path in (
        TABLES,
        FIGURES,
        PAPER_TABLES,
        PAPER_FIGURES,
        ACTIVE_PAPER_TABLES,
    ):
        path.mkdir(parents=True, exist_ok=True)


def save_table(
    frame: pd.DataFrame,
    stem: str,
    decimals: int = 4,
    column_format: str | None = None,
    table_width: str | None = None,
    *,
    escape: bool = True,
    formatters: dict[str, object] | None = None,
) -> None:
    frame.to_csv(TABLES / f"{stem}.csv", index=False)
    latex = frame.to_latex(
        index=False,
        escape=escape,
        na_rep="--",
        float_format=f"%.{decimals}f",
        column_format=column_format,
        formatters=formatters,
    )
    if table_width is not None:
        latex = latex.replace(
            r"\begin{tabular}",
            rf"\begin{{tabularx}}{{{table_width}}}",
            1,
        ).replace(r"\end{tabular}", r"\end{tabularx}", 1)
    lines = latex.splitlines()
    header_index = lines.index(r"\toprule") + 1
    header = lines[header_index].removesuffix(r" \\")
    lines[header_index] = (
        " & ".join(rf"\textbf{{{cell}}}" for cell in header.split(" & "))
        + r" \\"
    )
    latex = "\n".join(lines) + "\n"
    (PAPER_TABLES / f"{stem}.tex").write_text(latex, encoding="utf-8")
    if stem in ACTIVE_TABLE_STEMS:
        (ACTIVE_PAPER_TABLES / f"{stem}.tex").write_text(
            latex,
            encoding="utf-8",
        )


def weighted_table_format(
    first_column_weight: float,
    *centered_column_weights: float,
) -> str:
    """Return a full-width tabularx layout with one left and centered remainder."""
    weights = [first_column_weight, *centered_column_weights]
    if not np.isclose(sum(weights), len(weights), atol=1e-6):
        raise ValueError("Weighted tabularx columns must sum to their column count")
    first = f"U{{{first_column_weight:.6g}}}"
    centered = "".join(f"V{{{weight:.6g}}}" for weight in centered_column_weights)
    return f"@{{}}{first}{centered}@{{}}"


def external_table_panels(external: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the wide external-boundary table into lossless portrait panels."""
    identifiers = ["Dataset", "Weighting", "N occupations"]
    prediction_columns = identifiers + [
        "Joint profile p-value",
        "Delta R2",
        "Baseline grouped RMSE",
        "Profile grouped RMSE",
        "Grouped RMSE change",
        "Profile model out-of-fold R2",
    ]
    coefficient_columns = identifiers + [
        "Reuse coefficient",
        "Reuse HC1 SE",
        "Reuse p-value",
        "Effective count coefficient",
        "Effective count HC1 SE",
        "Effective count p-value",
    ]
    return external[prediction_columns].copy(), external[coefficient_columns].copy()


def exact_title_stable_soc_codes(
    reference: pd.DataFrame, later: pd.DataFrame
) -> set[str]:
    """Return common SOC codes whose release titles match after trivial cleanup."""
    required = {"soc", "occupation_title"}
    for label, frame in (("reference", reference), ("later", later)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{label} title frame is missing columns: {sorted(missing)}")
        if frame["soc"].duplicated().any():
            raise ValueError(f"{label} title frame must contain unique SOC codes")
    merged = reference[list(required)].merge(
        later[list(required)],
        on="soc",
        how="inner",
        suffixes=("_reference", "_later"),
        validate="one_to_one",
    )
    for suffix in ("reference", "later"):
        merged[f"normalized_title_{suffix}"] = (
            merged[f"occupation_title_{suffix}"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.casefold()
        )
    stable = merged["normalized_title_reference"].eq(
        merged["normalized_title_later"]
    )
    return set(merged.loc[stable, "soc"].astype(str))


def zscore(values: pd.Series) -> pd.Series:
    values = values.astype(float)
    scale = float(values.std(ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        return pd.Series(np.nan, index=values.index)
    return (values - float(values.mean())) / scale


def reconstruction_audit(occupations: pd.DataFrame, ledger: pd.DataFrame) -> dict[str, object]:
    residual = occupations["published_residual"].dropna().to_numpy(dtype=float)
    correlation = float(
        occupations[["standardized_score", "published_aioe"]].corr().iloc[0, 1]
    )
    weight_error = (
        ledger.groupby("soc")["effective_weight"].sum().sub(1.0).abs().max()
    )
    return {
        "accounted_score": "FRS score rebuilt from archived full-precision inputs",
        "published_comparator": "FRS Appendix A AIOE",
        "standardization_ddof": 1,
        "occupations": int(occupations.shape[0]),
        "abilities": int(ledger["Ability"].nunique()),
        "ledger_rows": int(ledger.shape[0]),
        "maximum_weight_sum_error": float(weight_error),
        "maximum_internal_identity_error": float(occupations["reconstruction_error"].abs().max()),
        "published_correlation": correlation,
        "published_rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "maximum_absolute_published_residual": float(np.max(np.abs(residual))),
        "exactness_boundary": (
            "Ability contributions exactly reconstruct the locally rebuilt score; "
            "the archived inputs do not reproduce published Appendix A bit-for-bit."
        ),
    }


def build_audit_table(audit: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Audit quantity": "Six-digit occupations", "Result": f"{audit['occupations']:,}"},
            {"Audit quantity": "Abilities per occupation", "Result": str(audit["abilities"])},
            {"Audit quantity": "Occupation-ability rows", "Result": f"{audit['ledger_rows']:,}"},
            {
                "Audit quantity": "Maximum internal identity error",
                "Result": (
                    rf"\({audit['maximum_internal_identity_error'] / 1e-15:.2f}"
                    r"\times 10^{-15}\)"
                ),
            },
            {
                "Audit quantity": r"Published-score Pearson \(r\)",
                "Result": f"{audit['published_correlation']:.6f}",
            },
            {
                "Audit quantity": "Published-score RMSE",
                "Result": f"{audit['published_rmse']:.5f}",
            },
            {
                "Audit quantity": "Maximum absolute published residual",
                "Result": f"{audit['maximum_absolute_published_residual']:.5f}",
            },
        ]
    )


def build_profile_summary(occupations: pd.DataFrame) -> pd.DataFrame:
    definitions = [
        ("Positive-profile reuse", "positive_reuse"),
        ("Effective ability count", "effective_ability_count"),
        ("Positive-profile HHI", "exposure_concentration"),
        ("Positive contribution mass", "positive_account"),
        ("Negative contribution mass", "negative_account"),
    ]
    rows = []
    for label, column in definitions:
        values = occupations[column].dropna()
        rows.append(
            {
                "Diagnostic": label,
                "N": int(values.shape[0]),
                "Mean": float(values.mean()),
                "SD": float(values.std(ddof=1)),
                "P10": float(values.quantile(0.10)),
                "Median": float(values.median()),
                "P90": float(values.quantile(0.90)),
                "Correlation with published AIOE": float(
                    occupations[[column, "published_aioe"]].corr().iloc[0, 1]
                ),
            }
        )
    return pd.DataFrame(rows)


def build_profile_summary_main(occupations: pd.DataFrame) -> pd.DataFrame:
    """Return the nonredundant profile summary used in the main manuscript."""
    full = build_profile_summary(occupations)
    keep = full[~full["Diagnostic"].eq("Positive-profile HHI")].copy()
    output = keep[
        [
            "Diagnostic",
            "Mean",
            "Median",
            "P90",
            "Correlation with published AIOE",
        ]
    ].reset_index(drop=True)
    return output.rename(
        columns={
            "Correlation with published AIOE": r"\(r_{\mathrm{AIOE}}\)"
        }
    )


def select_illustrative_pairs(
    pairs: pd.DataFrame,
    occupations: pd.DataFrame,
) -> pd.DataFrame:
    score = occupations.set_index("soc")["published_aioe"].dropna()
    breaks = score.quantile([0.25, 0.50, 0.75]).to_numpy(dtype=float)
    selected: list[pd.Series] = []
    used: set[str] = set()
    candidates = pairs.copy()
    candidates["pair_score"] = 0.5 * (candidates["score_left"] + candidates["score_right"])
    candidates["score_quartile"] = np.digitize(candidates["pair_score"], breaks) + 1
    for quartile in range(1, 5):
        subset = candidates[candidates["score_quartile"].eq(quartile)].sort_values(
            ["jensen_shannon_distance", "score_gap", "soc_left", "soc_right"],
            ascending=[False, True, True, True],
        )
        for _, row in subset.iterrows():
            if row["soc_left"] not in used and row["soc_right"] not in used:
                selected.append(row)
                used.update([row["soc_left"], row["soc_right"]])
                break
    output = pd.DataFrame(selected).reset_index(drop=True)
    if output.shape[0] != 4:
        raise ValueError("The deterministic pair rule did not produce one pair per score quartile")
    output.insert(
        0,
        "selection_rule",
        "maximum base-2 Jensen-Shannon distance within full 774-occupation AIOE quartile; "
        "published and rebuilt score gaps <= 0.025; cross-major; "
        "deterministic ties; no occupation reuse",
    )
    return output


def build_pair_eligible_occupations(
    occupations: pd.DataFrame,
    census_soc_crosswalk: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict to Census-mappable SOCs without consulting earnings outcomes."""
    if "soc" not in occupations.columns or "soc" not in census_soc_crosswalk.columns:
        raise ValueError("occupation and crosswalk inputs must contain soc")
    supported = set(census_soc_crosswalk["soc"].dropna().astype(str))
    return occupations[occupations["soc"].astype(str).isin(supported)].sort_values(
        "soc"
    ).reset_index(drop=True)


def build_distance_benchmark(
    pairs: pd.DataFrame,
    drift: pd.DataFrame,
    later_drift: pd.DataFrame | None = None,
) -> pd.DataFrame:
    matched_values = pairs["jensen_shannon_distance"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []

    def summarize(
        label: str,
        values: np.ndarray,
        pair_share: float | None,
    ) -> None:
        rows.append(
            {
                "Comparison": label,
                "N": int(values.size),
                "Exact-zero share": float(np.mean(values == 0.0)),
                "Mean base-2 Jensen-Shannon distance": float(np.mean(values)),
                "Median base-2 Jensen-Shannon distance": float(np.median(values)),
                "P90 base-2 Jensen-Shannon distance": float(
                    np.quantile(values, 0.90)
                ),
                "Pair share above benchmark P90": (
                    float(pair_share) if pair_share is not None else np.nan
                ),
            }
        )

    summarize("Near-score cross-major occupation pairs", matched_values, None)
    for current in [drift, later_drift]:
        if current is None:
            continue
        if "any_ability_weight_updated" not in current.columns:
            raise ValueError(
                "version drift benchmark requires any_ability_weight_updated"
            )
        earlier = str(current["earlier_release"].iloc[0])
        later = str(current["later_release"].iloc[0])
        for suffix, sample in [
            ("all crosswalk-supported matches", current),
            (
                "updated weights only",
                current[current["any_ability_weight_updated"].astype(bool)],
            ),
        ]:
            values = sample["jensen_shannon_distance"].to_numpy(dtype=float)
            if values.size == 0:
                raise ValueError(
                    f"version drift sample {earlier} to {later} ({suffix}) is empty"
                )
            p90 = float(np.quantile(values, 0.90))
            summarize(
                f"Same occupation, O*NET {earlier} to {later} ({suffix})",
                values,
                float(np.mean(matched_values > p90)),
            )
    return pd.DataFrame(rows)


def build_caliper_sensitivity(
    ledger: pd.DataFrame,
    occupations: pd.DataFrame,
    drift: pd.DataFrame,
    later_drift: pd.DataFrame,
    calipers: tuple[float, ...] = (0.010, 0.025, 0.050),
) -> pd.DataFrame:
    """Re-estimate the pair benchmark across calipers and major-group rules."""
    required_drift = {"jensen_shannon_distance", "any_ability_weight_updated"}
    for label, frame in (("27.1", drift), ("30.3", later_drift)):
        missing = required_drift.difference(frame.columns)
        if missing:
            raise ValueError(
                f"{label} version drift is missing columns: {sorted(missing)}"
            )
    if not calipers or any(not np.isfinite(value) or value < 0 for value in calipers):
        raise ValueError("calipers must contain finite nonnegative values")

    metadata = occupations.copy()
    if "occupation_title" not in metadata.columns:
        metadata["occupation_title"] = metadata["soc"].astype(str)
    updated_27 = drift[drift["any_ability_weight_updated"].astype(bool)]
    updated_30 = later_drift[
        later_drift["any_ability_weight_updated"].astype(bool)
    ]
    if updated_27.empty or updated_30.empty:
        raise ValueError("updated-only version-drift benchmarks must be nonempty")
    p90_27 = float(updated_27["jensen_shannon_distance"].quantile(0.90))
    p90_30 = float(updated_30["jensen_shannon_distance"].quantile(0.90))

    rows: list[dict[str, object]] = []
    for require_different_major, pair_design in [
        (True, "Cross-major"),
        (False, "All eligible pairs"),
    ]:
        for caliper in calipers:
            pairs = build_profile_distance_table(
                ledger,
                metadata,
                score_col="published_aioe",
                secondary_score_col="standardized_score",
                tolerance=float(caliper),
                require_different_major=require_different_major,
            )
            if pairs.empty:
                raise ValueError(
                    f"caliper sensitivity produced no pairs for {pair_design} "
                    f"at {caliper:.3f}"
                )
            distances = pairs["jensen_shannon_distance"].astype(float)
            rows.append(
                {
                    "Pair design": pair_design,
                    "Caliper": float(caliper),
                    "Eligible pairs": int(pairs.shape[0]),
                    "Median JSD": float(distances.median()),
                    "Share above 27.1 P90": float((distances > p90_27).mean()),
                    "Share above 30.3 P90": float((distances > p90_30).mean()),
                }
            )
    return pd.DataFrame(rows)


def _boundary_formula(include_profiles: bool) -> str:
    terms = (
        "bs(aioe, df=4, degree=3) + Q('Job Zone') + edu_score + "
        "pretrend_wage + C(major_group)"
    )
    if include_profiles:
        terms += " + positive_reuse_std + effective_ability_count_std"
    return f"outcome_change ~ {terms}"


def _prediction_pipeline(include_profiles: bool) -> Pipeline:
    numeric = ["Job Zone", "edu_score", "pretrend_wage"]
    if include_profiles:
        numeric += ["positive_reuse", "effective_ability_count"]
    transformer = ColumnTransformer(
        [
            (
                "aioe_spline",
                Pipeline(
                    [
                        ("spline", SplineTransformer(n_knots=4, degree=3, include_bias=False)),
                        ("scale", StandardScaler()),
                    ]
                ),
                ["aioe"],
            ),
            ("numeric", StandardScaler(), numeric),
            ("major", OneHotEncoder(handle_unknown="ignore"), ["major_group"]),
        ]
    )
    return Pipeline([("features", transformer), ("model", LinearRegression())])


def grouped_prediction_increment(
    sample: pd.DataFrame, weight_col: str | None
) -> tuple[float, float, float, float]:
    required = [
        "outcome_change",
        "aioe",
        "Job Zone",
        "edu_score",
        "pretrend_wage",
        "major_group",
        "positive_reuse",
        "effective_ability_count",
    ]
    data = sample.dropna(subset=required).reset_index(drop=True)
    splits = list(LeaveOneGroupOut().split(data, groups=data["major_group"]))
    metrics = {}
    for include_profiles, label in [(False, "baseline"), (True, "profiles")]:
        predicted = np.full(data.shape[0], np.nan)
        for train, test in splits:
            model = _prediction_pipeline(include_profiles)
            fit_kwargs = {}
            if weight_col is not None:
                fit_kwargs["model__sample_weight"] = data.loc[train, weight_col]
            model.fit(data.loc[train], data.loc[train, "outcome_change"], **fit_kwargs)
            predicted[test] = model.predict(data.loc[test])
        truth = data["outcome_change"].to_numpy(dtype=float)
        evaluation_weights = (
            data[weight_col].to_numpy(dtype=float) if weight_col is not None else None
        )
        metrics[label] = (
            float(
                np.sqrt(
                    mean_squared_error(
                        truth, predicted, sample_weight=evaluation_weights
                    )
                )
            ),
            float(r2_score(truth, predicted, sample_weight=evaluation_weights)),
        )
    return (
        metrics["baseline"][0],
        metrics["profiles"][0],
        metrics["profiles"][0] - metrics["baseline"][0],
        metrics["profiles"][1],
    )


def fit_external_boundary(
    sample: pd.DataFrame,
    dataset: str,
    weighting: str,
    weight_col: str | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    required = [
        "outcome_change",
        "aioe",
        "Job Zone",
        "edu_score",
        "pretrend_wage",
        "major_group",
        *PROFILE_TERMS,
    ]
    data = sample.dropna(subset=required).copy()
    if weight_col is None:
        base = smf.ols(_boundary_formula(False), data=data).fit()
        full_plain = smf.ols(_boundary_formula(True), data=data).fit()
        full = smf.ols(_boundary_formula(True), data=data).fit(cov_type="HC1")
    else:
        base = smf.wls(_boundary_formula(False), data=data, weights=data[weight_col]).fit()
        full_plain = smf.wls(_boundary_formula(True), data=data, weights=data[weight_col]).fit()
        full = smf.wls(_boundary_formula(True), data=data, weights=data[weight_col]).fit(cov_type="HC1")
    baseline_rmse, profile_rmse, rmse_delta, profile_oof_r2 = grouped_prediction_increment(
        data, weight_col
    )
    output: dict[str, object] = {
        "Dataset": dataset,
        "Weighting": weighting,
        "N occupations": int(data.shape[0]),
        "Joint profile p-value": robust_joint_pvalue(full, PROFILE_TERMS),
        "Delta R2": float(full_plain.rsquared - base.rsquared),
        "Baseline grouped RMSE": baseline_rmse,
        "Profile grouped RMSE": profile_rmse,
        "Grouped RMSE change": rmse_delta,
        "Profile model out-of-fold R2": profile_oof_r2,
    }
    registry = []
    for term in PROFILE_TERMS:
        estimate, standard_error, pvalue = coefficient(full, term)
        prefix = "Reuse" if term.startswith("positive") else "Effective count"
        output[f"{prefix} coefficient"] = estimate
        output[f"{prefix} HC1 SE"] = standard_error
        output[f"{prefix} p-value"] = pvalue
        registry.append(
            {
                "family": "two_layer_external_association",
                "dataset": dataset,
                "weighting": weighting,
                "test": PROFILE_LABELS[term],
                "p_value": pvalue,
            }
        )
    registry.append(
        {
            "family": "two_layer_external_association_joint",
            "dataset": dataset,
            "weighting": weighting,
            "test": "Two occupation-level profile summaries (G+ and Neff)",
            "p_value": output["Joint profile p-value"],
        }
    )
    return output, registry


def build_external_associations(
    profiles: pd.DataFrame,
    title_stable_soc: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    profile_measures = profiles.copy()
    profile_measures["positive_reuse_std"] = zscore(profile_measures["positive_reuse"])
    profile_measures["effective_ability_count_std"] = zscore(
        profile_measures["effective_ability_count"]
    )
    controls = build_measure_frame("27_1")[[
        "soc",
        "Job Zone",
        "edu_score",
        "major_group",
    ]]
    measures = profile_measures.merge(controls, on="soc", how="inner", validate="one_to_one")
    measures = measures.rename(columns={"published_aioe": "aioe"})
    panel = pd.read_csv(RESULTS / "panel_annual.csv", dtype={"soc": str})
    cells = pd.read_csv(MASTER / "acs_occupation_year_cells.csv", dtype={"soc": str})
    cells["ln_incwage_2024"] = deflate_log_values(
        cells["ln_incwage"], cells["YEAR"], CPI_U_ANNUAL, base_year=2024
    )
    oews = build_oe_change(panel, measures)
    acs = build_acs_change(cells, measures, outcome_col="ln_incwage_2024").merge(
        oews[["soc", "pretrend_wage"]], on="soc", how="left", validate="one_to_one"
    )
    rows = []
    registry = []
    specifications = [
        (acs, "ACS annual earnings", "Positive-earner weighted", "analysis_weight"),
        (acs, "ACS annual earnings", "Equal occupation", None),
        (oews, "OEWS published estimates", "Equal occupation", None),
    ]
    if title_stable_soc is not None:
        acs_stable = acs[acs["soc"].isin(title_stable_soc)].copy()
        oews_stable = oews[oews["soc"].isin(title_stable_soc)].copy()
        specifications.extend(
            [
                (
                    acs_stable,
                    "ACS annual earnings; title-stable codes",
                    "Positive-earner weighted",
                    "analysis_weight",
                ),
                (
                    acs_stable,
                    "ACS annual earnings; title-stable codes",
                    "Equal occupation",
                    None,
                ),
                (
                    oews_stable,
                    "OEWS published estimates; title-stable codes",
                    "Equal occupation",
                    None,
                ),
            ]
        )
    for sample, dataset, weighting, weight_col in specifications:
        row, tests = fit_external_boundary(sample, dataset, weighting, weight_col)
        rows.append(row)
        registry.extend(tests)
    counts = {
        "acs_person_records": int(cells["n"].sum()),
        "acs_occupations": int(acs.shape[0]),
        "oews_occupations": int(oews.shape[0]),
    }
    if title_stable_soc is not None:
        counts.update(
            {
                "title_stable_acs_mapped_occupations": int(acs_stable.shape[0]),
                "title_stable_oews_mapped_occupations": int(oews_stable.shape[0]),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(registry), counts


def plot_framework(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 3.55))
    ax.set_axis_off()
    boxes = [
        (
            0.03,
            0.62,
            0.20,
            0.24,
            "Archived FRS inputs\nAbility ratings, application map,\noccupational aggregation",
            "#E8F0F7",
        ),
        (
            0.29,
            0.62,
            0.25,
            0.24,
            "Layer 1: Exact accounting\n52 signed ability contributions\nExact local-score reconstruction",
            "#BCD7E9",
        ),
        (
            0.60,
            0.62,
            0.35,
            0.24,
            "Reconciliation boundary\nPublished score compared with rebuilt score\nResidual reported, never reallocated",
            "#F3E4C7",
        ),
        (
            0.29,
            0.17,
            0.25,
            0.25,
            "Layer 2: Profile diagnostics\nReuse, concentration,\nand ability-contribution profile distance",
            "#D9E4ED",
        ),
        (
            0.60,
            0.17,
            0.35,
            0.25,
            "Permitted use\nMeasurement audit and\noccupational technology assessment",
            "#E6E6E6",
        ),
    ]
    for index, (x, y, width, height, label, color) in enumerate(boxes):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                facecolor=color,
                edgecolor="#000000",
                linewidth=0.75 if index == 2 else 1.0,
                linestyle="--" if index == 2 else "-",
            )
        )
        ax.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=8.3,
            linespacing=1.08,
        )
    arrows = [((0.23, 0.74), (0.29, 0.74)), ((0.54, 0.74), (0.60, 0.74)), ((0.415, 0.62), (0.415, 0.42)), ((0.54, 0.295), (0.60, 0.295))]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.25, "color": "#000000"})
    ax.text(
        0.03,
        0.96,
        "Two-layer occupational AI exposure audit",
        fontsize=10.2,
        weight="bold",
        va="top",
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_distance_benchmark(
    pairs: pd.DataFrame,
    drift: pd.DataFrame,
    later_drift: pd.DataFrame,
    output: Path,
    *,
    compact: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.55) if compact else (6.8, 4.2))
    updated = drift[drift["any_ability_weight_updated"].astype(bool)]
    later_updated = later_drift[
        later_drift["any_ability_weight_updated"].astype(bool)
    ]
    for frame, label, color, linestyle in [
        (pairs, "Near-score pairs", "#0072B2", "-"),
        (updated, "O*NET 24.3–27.1 updates", "#D55E00", "--"),
        (later_updated, "O*NET 24.3–30.3 updates", "#000000", ":"),
    ]:
        values = np.sort(frame["jensen_shannon_distance"].to_numpy(dtype=float))
        probability = np.arange(1, values.size + 1) / values.size
        ax.step(
            values,
            probability,
            where="post",
            label=f"{label} (N={values.size:,})",
            color=color,
            linestyle=linestyle,
            lw=1.55 if compact else 2.0,
        )
    ax.set_xlabel(
        "Base-2 Jensen-Shannon distance\n(52-ability contribution-mass profile)"
        if compact
        else "Base-2 Jensen-Shannon distance (52-ability contribution-mass profile)",
        fontsize=7.5 if compact else 9,
    )
    ax.set_ylabel(
        "Cumulative probability",
        fontsize=7.5 if compact else 9,
    )
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.01)
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.tick_params(labelsize=7 if compact else 8)
    ax.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.08),
        fontsize=6.6 if compact else 8,
        handlelength=2.4,
        labelspacing=0.35,
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_matched_profiles(ledger: pd.DataFrame, pairs: pd.DataFrame, output: Path) -> None:
    target = pairs[
        pairs["occupation_left"].str.contains("Rehabilitation Counselors", case=False)
        & pairs["occupation_right"].str.contains("Optometrists", case=False)
    ]
    if target.empty:
        target = pairs[
            pairs["occupation_right"].str.contains(
                "Rehabilitation Counselors", case=False
            )
            & pairs["occupation_left"].str.contains("Optometrists", case=False)
        ]
    if target.empty:
        raise ValueError(
            "the illustrative Rehabilitation Counselors–Optometrists pair was not selected"
        )
    pair = target.iloc[0]
    left = ledger[ledger["soc"].eq(pair["soc_left"])].set_index("Ability")
    right = ledger[ledger["soc"].eq(pair["soc_right"])].set_index("Ability")
    combined = pd.DataFrame(
        {
            "left": left["standardized_contribution"],
            "right": right["standardized_contribution"],
        }
    ).fillna(0.0)
    combined["combined_absolute"] = combined["left"].abs() + combined["right"].abs()
    abilities = (
        combined.nlargest(12, "combined_absolute")
        .sort_values("combined_absolute")
        .index.tolist()
    )
    fig, ax = plt.subplots(figsize=(3.35, 4.15))
    y = np.arange(len(abilities), dtype=float)
    height = 0.34
    left_values = combined.loc[abilities, "left"].to_numpy(dtype=float)
    right_values = combined.loc[abilities, "right"].to_numpy(dtype=float)
    left_bars = ax.barh(
        y - height / 2,
        left_values,
        height=height,
        color="#0072B2",
        alpha=1.0,
        label=f"{pair['occupation_left']}\n(AIOE {pair['score_left']:.3f})",
    )
    right_bars = ax.barh(
        y + height / 2,
        right_values,
        height=height,
        color="#D55E00",
        alpha=1.0,
        label=f"{pair['occupation_right']}\n(AIOE {pair['score_right']:.3f})",
    )
    for bars, values in ((left_bars, left_values), (right_bars, right_values)):
        for bar, value in zip(bars, values, strict=True):
            if value < 0:
                bar.set_hatch("//")
    wrapped_abilities = [textwrap.fill(ability, width=20) for ability in abilities]
    ax.set_yticks(y, wrapped_abilities)
    ax.tick_params(axis="y", labelsize=6.8, pad=2)
    ax.tick_params(axis="x", labelsize=7)
    ax.axvline(0, color="#000000", lw=0.9)
    ax.set_xlabel("Signed contribution to reconstructed score", fontsize=7.5)
    ax.text(
        0.5,
        1.015,
        f"Full-profile JSD = {pair['jensen_shannon_distance']:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=7.5,
    )
    ax.grid(axis="x", alpha=0.18, linewidth=0.6)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.055),
        ncol=2,
        fontsize=7.5,
        columnspacing=0.9,
        handlelength=1.6,
    )
    limit = float(
        np.max(np.abs(combined.loc[abilities, ["left", "right"]].to_numpy()))
    )
    ax.set_xlim(-1.08 * limit, 1.08 * limit)
    fig.tight_layout(pad=0.45)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_directories()
    ledger, occupations = build_frs_replication_accounting()
    audit = reconstruction_audit(occupations, ledger)
    audit.update(
        {
            "formulas": {
                "raw_scalar": "(IM/5)*(LV/7)",
                "effective_weight": "sum_d(raw_scalar_doa)/sum_b sum_d(raw_scalar_dob)",
                "unstandardized_score": "sum_a(effective_weight_oa*ability_score_a)",
                "standardized_score": "(unstandardized_score_o-mean_R)/sample_sd_R",
                "ability_contribution": "effective_weight_oa*(ability_score_a-mean_R)/sample_sd_R",
                "internal_identity": "standardized_score_o=sum_a(contribution_oa)=positive_account_o-negative_account_o",
                "published_reconciliation": "published_AIOE_o=standardized_score_o+published_residual_o",
            },
            "standardization_universe": {
                "occupations": 774,
                "occupation_level": "six-digit SOC",
                "abilities_per_occupation": 52,
                "reference_set": "archived FRS occupation universe",
                "standard_deviation_ddof": 1,
            },
            "aggregation_order": (
                "aggregate detailed-occupation raw scalars to six-digit SOC, "
                "then normalize once within six-digit occupation"
            ),
            "acceptance_gates": {
                "maximum_weight_sum_error": 1e-12,
                "maximum_internal_identity_error": 1e-12,
                "minimum_published_correlation": 0.99999,
                "maximum_published_rmse": 0.00360,
                "maximum_absolute_published_residual": 0.038,
            },
            "gate_purpose": (
                "Executable regression thresholds around the archived reconstruction; "
                "not preregistered evidentiary or inferential cutoffs."
            ),
            "audit_metric_stage": (
                "Audit errors are computed from in-memory full-precision frames before CSV "
                "serialization; reloading decimal CSV text can introduce machine-scale "
                "round-off without changing any acceptance gate."
            ),
        }
    )
    ledger.to_csv(RESULTS / "exact_ability_accounting.csv", index=False)
    occupations.to_csv(RESULTS / "occupation_accounting_summary.csv", index=False)
    (RESULTS / "reconstruction_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    save_table(
        build_audit_table(audit),
        "table_exact_accounting_audit",
        decimals=6,
        column_format=weighted_table_format(1.25, 0.75),
        table_width=r"\linewidth",
        escape=False,
    )
    profile_summary = build_profile_summary(occupations).rename(
        columns={
            "Correlation with published AIOE": (
                r"Pearson \(r\) with published AIOE"
            )
        }
    )
    save_table(
        profile_summary,
        "table_profile_summary",
        decimals=4,
        column_format=weighted_table_format(
            1.70,
            0.90,
            0.90,
            0.90,
            0.90,
            0.90,
            0.90,
            0.90,
        ),
        table_width=r"\linewidth",
        escape=False,
    )
    save_table(
        build_profile_summary_main(occupations),
        "table_profile_summary_main",
        decimals=4,
        column_format=weighted_table_format(1.80, 0.80, 0.80, 0.80, 0.80),
        table_width=r"\linewidth",
        escape=False,
    )

    crosswalk = load_census_soc_crosswalk()
    pair_occupations = build_pair_eligible_occupations(occupations, crosswalk)
    pair_soc = set(pair_occupations["soc"])
    pairs = build_profile_distance_table(
        ledger,
        pair_occupations,
        score_col="published_aioe",
        secondary_score_col="standardized_score",
        tolerance=0.025,
        require_different_major=True,
    )
    if pairs.empty:
        raise ValueError("The declared score-band rule produced no matched occupation pairs")
    pairs.to_csv(RESULTS / "within_score_profile_distances.csv", index=False)
    selected = select_illustrative_pairs(pairs, occupations)
    selected.to_csv(RESULTS / "illustrative_pair_selection_manifest.csv", index=False)
    pair_table = selected[[
        "score_quartile",
        "occupation_left",
        "score_left",
        "occupation_right",
        "score_right",
        "score_gap",
        "secondary_score_gap",
        "jensen_shannon_distance",
    ]].rename(
        columns={
            "score_quartile": "AIOE quartile",
            "occupation_left": "Occupation A",
            "score_left": "AIOE A",
            "occupation_right": "Occupation B",
            "score_right": "AIOE B",
            "score_gap": "Absolute score gap",
            "secondary_score_gap": "Rebuilt-score gap",
            "jensen_shannon_distance": "Base-2 Jensen-Shannon distance",
        }
    )
    save_table(
        pair_table,
        "table_illustrative_profile_pairs",
        decimals=4,
        column_format=weighted_table_format(
            0.70,
            1.50,
            0.70,
            1.50,
            0.70,
            0.90,
            0.90,
            1.10,
        ),
        table_width=r"\linewidth",
    )

    ledger_243, occupations_243 = build_onet_release_accounting("24_3")
    center = float(ledger_243["score_center"].iloc[0])
    scale = float(ledger_243["score_scale"].iloc[0])
    ledger_271, occupations_271 = build_onet_release_accounting(
        "27_1", reference_center=center, reference_scale=scale
    )
    ledger_303, occupations_303 = build_onet_release_accounting(
        "30_3", reference_center=center, reference_scale=scale
    )
    drift = build_version_drift_table(ledger_243, ledger_271, "24.3", "27.1")
    drift = drift[drift["soc"].isin(pair_soc)].reset_index(drop=True)
    drift.to_csv(RESULTS / "version_profile_drift.csv", index=False)
    later_drift = build_version_drift_table(
        ledger_243, ledger_303, "24.3", "30.3"
    )
    later_drift = later_drift[
        later_drift["soc"].isin(pair_soc)
    ].reset_index(drop=True)
    later_drift.to_csv(
        RESULTS / "version_profile_drift_24_3_to_30_3.csv",
        index=False,
    )
    version_title_stable_soc = exact_title_stable_soc_codes(
        occupations_243, occupations_271
    )
    later_version_title_stable_soc = exact_title_stable_soc_codes(
        occupations_243, occupations_303
    )
    title_stable_drift = drift[
        drift["soc"].isin(version_title_stable_soc)
    ].reset_index(drop=True)
    title_stable_drift_p90 = float(
        title_stable_drift["jensen_shannon_distance"].quantile(0.90)
    )
    updated_drift = drift[
        drift["any_ability_weight_updated"].astype(bool)
    ].reset_index(drop=True)
    later_updated_drift = later_drift[
        later_drift["any_ability_weight_updated"].astype(bool)
    ].reset_index(drop=True)
    title_stable_updated_drift = updated_drift[
        updated_drift["soc"].isin(version_title_stable_soc)
    ].reset_index(drop=True)
    later_title_stable_updated_drift = later_updated_drift[
        later_updated_drift["soc"].isin(later_version_title_stable_soc)
    ].reset_index(drop=True)

    caliper_sensitivity = build_caliper_sensitivity(
        ledger,
        pair_occupations,
        drift,
        later_drift,
    )
    caliper_sensitivity.to_csv(
        RESULTS / "profile_caliper_sensitivity.csv",
        index=False,
    )
    caliper_display = caliper_sensitivity.copy()
    caliper_display["Caliper"] = caliper_display["Caliper"].map(
        lambda value: f"{value:.3f}"
    )
    caliper_display["Eligible pairs"] = caliper_display["Eligible pairs"].map(
        lambda value: f"{int(value):,}"
    )
    caliper_display["Median JSD"] = caliper_display["Median JSD"].map(
        lambda value: f"{value:.4f}"
    )
    for column in ["Share above 27.1 P90", "Share above 30.3 P90"]:
        caliper_display[column] = caliper_display[column].map(
            lambda value: f"{100 * value:.1f}\\%"
        )
    caliper_display = caliper_display.rename(
        columns={
            "Share above 27.1 P90": "Share above 27.1 updated-only P90",
            "Share above 30.3 P90": "Share above 30.3 updated-only P90",
        }
    )
    save_table(
        caliper_display,
        "table_profile_caliper_sensitivity",
        decimals=4,
        column_format=weighted_table_format(
            1.40,
            0.65,
            0.65,
            0.80,
            1.25,
            1.25,
        ),
        table_width=r"\linewidth",
        escape=False,
    )

    def title_sensitivity_row(
        release_comparison: str,
        screen: str,
        sample: pd.DataFrame,
    ) -> dict[str, object]:
        distances = sample["jensen_shannon_distance"]
        p90 = float(distances.quantile(0.90))
        return {
            "Release comparison": release_comparison,
            "Version-drift screen": screen,
            "Drift occupations": int(sample.shape[0]),
            "Median base-2 Jensen-Shannon distance": float(distances.median()),
            "P90 base-2 Jensen-Shannon distance": p90,
            "Pair share above drift P90": float(
                (pairs["jensen_shannon_distance"] > p90).mean()
            ),
        }

    title_sensitivity = pd.DataFrame(
        [
            title_sensitivity_row(
                "O*NET 24.3 to 27.1",
                "Updated weights only",
                updated_drift,
            ),
            title_sensitivity_row(
                "O*NET 24.3 to 27.1",
                "Updated weights and unchanged title",
                title_stable_updated_drift,
            ),
            title_sensitivity_row(
                "O*NET 24.3 to 30.3",
                "Updated weights only",
                later_updated_drift,
            ),
            title_sensitivity_row(
                "O*NET 24.3 to 30.3",
                "Updated weights and unchanged title",
                later_title_stable_updated_drift,
            ),
        ]
    )
    save_table(
        title_sensitivity,
        "table_profile_title_stability_sensitivity",
        decimals=4,
        column_format=weighted_table_format(1.00, 1.60, 0.70, 0.90, 0.90, 0.90),
        table_width=r"\linewidth",
    )
    benchmark = build_distance_benchmark(pairs, drift, later_drift)
    save_table(
        benchmark,
        "table_profile_distance_benchmark",
        decimals=4,
        column_format=weighted_table_format(
            1.60,
            0.50,
            0.80,
            1.00,
            1.00,
            0.90,
            1.20,
        ),
        table_width=r"\linewidth",
    )
    benchmark_main_rows: list[dict[str, object]] = []
    benchmark_labels = {
        0: "Near-score occupation pairs",
        2: "O*NET 24.3--27.1, updated only",
        4: "O*NET 24.3--30.3, updated only",
    }
    for index, row in benchmark.iloc[[0, 2, 4]].iterrows():
        benchmark_main_rows.append(
            {
                "Benchmark": benchmark_labels[index],
                r"\(N\)": f"{int(row['N']):,}",
                "Zero share": (
                    "--"
                    if index == 0
                    else f"{100 * row['Exact-zero share']:.1f}\\%"
                ),
                "Median JSD": (
                    f"{row['Median base-2 Jensen-Shannon distance']:.4f}"
                ),
                "P90 JSD": f"{row['P90 base-2 Jensen-Shannon distance']:.4f}",
                "Near-score pairs above P90": (
                    "--"
                    if index == 0
                    else (
                        f"{100 * row['Pair share above benchmark P90']:.1f}\\%"
                    )
                ),
            }
        )
    save_table(
        pd.DataFrame(benchmark_main_rows),
        "table_profile_distance_benchmark_main",
        decimals=4,
        column_format=weighted_table_format(1.62, 0.55, 0.75, 0.90, 0.80, 1.38),
        table_width=r"\linewidth",
        escape=False,
    )

    stability_rows = []
    for label, current in [("27.1", occupations_271), ("30.3", occupations_303)]:
        merged = occupations_243.merge(current, on="soc", suffixes=("_243", "_later"))
        stability_rows.append(
            {
                "Later O*NET release": label,
                "Common occupations": int(merged.shape[0]),
                "Rebuilt score rank rho": float(spearmanr(merged["standardized_score_243"], merged["standardized_score_later"]).statistic),
                "Positive reuse rank rho": float(spearmanr(merged["positive_reuse_243"], merged["positive_reuse_later"]).statistic),
                "Effective count rank rho": float(spearmanr(merged["effective_ability_count_243"], merged["effective_ability_count_later"]).statistic),
            }
        )
    save_table(
        pd.DataFrame(stability_rows),
        "table_profile_version_stability",
        decimals=4,
        column_format=weighted_table_format(1.10, 0.85, 1.00, 1.00, 1.05),
        table_width=r"\linewidth",
    )

    external_title_stable_soc = exact_title_stable_soc_codes(
        occupations[["soc", "occupation_title"]], occupations_271
    )
    external, registry, counts = build_external_associations(
        occupations, title_stable_soc=external_title_stable_soc
    )
    save_table(external, "table_profile_external_association", decimals=4)
    external_prediction, external_coefficients = external_table_panels(external)
    save_table(
        external_prediction,
        "table_profile_external_association_prediction",
        decimals=4,
        column_format=weighted_table_format(
            1.60,
            1.30,
            0.60,
            0.80,
            0.70,
            1.00,
            1.00,
            1.00,
            1.00,
        ),
        table_width=r"\linewidth",
        formatters={
            "Delta R2": lambda value: f"{value:.6f}",
            "Baseline grouped RMSE": lambda value: f"{value:.6f}",
            "Profile grouped RMSE": lambda value: f"{value:.6f}",
            "Grouped RMSE change": lambda value: f"{value:.6f}",
        },
    )
    save_table(
        external_coefficients,
        "table_profile_external_association_coefficients",
        decimals=4,
        column_format=weighted_table_format(
            1.60,
            1.30,
            0.60,
            0.916667,
            0.916667,
            0.916667,
            0.916667,
            0.916667,
            0.916665,
        ),
        table_width=r"\linewidth",
    )
    external_main_rows: list[dict[str, object]] = []
    external_main = external[
        ~external["Dataset"].str.contains("title-stable", regex=False)
    ]
    for _, row in external_main.iterrows():
        dataset = (
            "ACS annual earnings"
            if row["Dataset"] == "ACS annual earnings"
            else "OEWS estimates"
        )
        rmse_change = float(row["Grouped RMSE change"])
        external_main_rows.append(
            {
                "Dataset": "ACS" if dataset == "ACS annual earnings" else "OEWS",
                "Weighting": row["Weighting"].replace(" weighted", ""),
                r"\(N\)": f"{int(row['N occupations']):,}",
                r"Joint \(p\)": f"{row['Joint profile p-value']:.4f}",
                r"\(\Delta R^2\)": f"{row['Delta R2']:.5f}",
                "Baseline RMSE": f"{row['Baseline grouped RMSE']:.6f}",
                r"\(\Delta\)RMSE": (
                    (
                        rf"\(-\){abs(rmse_change):.6f}"
                        if rmse_change < 0
                        else f"+{rmse_change:.6f}"
                    )
                ),
            }
        )
    save_table(
        pd.DataFrame(external_main_rows),
        "table_profile_external_association_main",
        decimals=4,
        column_format=weighted_table_format(
            1.00,
            1.30,
            0.55,
            0.75,
            1.10,
            1.10,
            1.20,
        ),
        table_width=r"\linewidth",
        escape=False,
    )
    registry.to_csv(RESULTS / "statistical_test_registry_two_layer.csv", index=False)

    plot_framework(PAPER_FIGURES / "two_layer_framework.pdf")
    plot_framework(FIGURES / "two_layer_framework.pdf")
    plot_distance_benchmark(
        pairs,
        drift,
        later_drift,
        PAPER_FIGURES / "profile_distance_benchmark_main.pdf",
        compact=True,
    )
    plot_distance_benchmark(
        pairs,
        drift,
        later_drift,
        PAPER_FIGURES / "profile_distance_benchmark.pdf",
    )
    plot_distance_benchmark(
        pairs,
        drift,
        later_drift,
        FIGURES / "profile_distance_benchmark.pdf",
    )
    plot_matched_profiles(ledger, selected, PAPER_FIGURES / "matched_occupation_profiles.pdf")
    plot_matched_profiles(ledger, selected, FIGURES / "matched_occupation_profiles.pdf")

    metrics = {
        **audit,
        **counts,
        "matched_pair_rule": (
            "Census-SOC crosswalk support; published and rebuilt score gaps <= 0.025; "
            "different two-digit SOC groups"
        ),
        "pair_candidate_occupations": int(pair_occupations.shape[0]),
        "pair_endpoint_occupations": int(
            len(set(pairs["soc_left"]) | set(pairs["soc_right"]))
        ),
        "matched_pairs": int(pairs.shape[0]),
        "version_drift_occupations": int(drift.shape[0]),
        "version_drift_exact_zero_share": float(
            (drift["jensen_shannon_distance"] == 0.0).mean()
        ),
        "version_drift_updated_occupations": int(
            drift["any_ability_weight_updated"].sum()
        ),
        "version_drift_updated_median_contribution_jsd": float(
            drift.loc[
                drift["any_ability_weight_updated"],
                "jensen_shannon_distance",
            ].median()
        ),
        "version_drift_updated_p90_contribution_jsd": float(
            drift.loc[
                drift["any_ability_weight_updated"],
                "jensen_shannon_distance",
            ].quantile(0.90)
        ),
        "matched_pair_share_above_updated_version_drift_p90": float(
            (
                pairs["jensen_shannon_distance"]
                > drift.loc[
                    drift["any_ability_weight_updated"],
                    "jensen_shannon_distance",
                ].quantile(0.90)
            ).mean()
        ),
        "later_version_drift_occupations": int(later_drift.shape[0]),
        "later_version_drift_exact_zero_share": float(
            (later_drift["jensen_shannon_distance"] == 0.0).mean()
        ),
        "later_version_drift_updated_occupations": int(
            later_drift["any_ability_weight_updated"].sum()
        ),
        "later_version_drift_updated_median_contribution_jsd": float(
            later_drift.loc[
                later_drift["any_ability_weight_updated"],
                "jensen_shannon_distance",
            ].median()
        ),
        "later_version_drift_updated_p90_contribution_jsd": float(
            later_drift.loc[
                later_drift["any_ability_weight_updated"],
                "jensen_shannon_distance",
            ].quantile(0.90)
        ),
        "matched_pair_share_above_later_updated_version_drift_p90": float(
            (
                pairs["jensen_shannon_distance"]
                > later_drift.loc[
                    later_drift["any_ability_weight_updated"],
                    "jensen_shannon_distance",
                ].quantile(0.90)
            ).mean()
        ),
        "title_stable_version_drift_occupations": int(title_stable_drift.shape[0]),
        "title_stable_version_drift_p90": title_stable_drift_p90,
        "matched_pair_share_above_title_stable_drift_p90": float(
            (pairs["jensen_shannon_distance"] > title_stable_drift_p90).mean()
        ),
        "title_stable_updated_version_drift_occupations": int(
            title_stable_updated_drift.shape[0]
        ),
        "title_stable_updated_version_drift_median": float(
            title_stable_updated_drift["jensen_shannon_distance"].median()
        ),
        "title_stable_updated_version_drift_p90": float(
            title_stable_updated_drift["jensen_shannon_distance"].quantile(0.90)
        ),
        "matched_pair_share_above_title_stable_updated_drift_p90": float(
            (
                pairs["jensen_shannon_distance"]
                > title_stable_updated_drift[
                    "jensen_shannon_distance"
                ].quantile(0.90)
            ).mean()
        ),
        "later_title_stable_updated_version_drift_occupations": int(
            later_title_stable_updated_drift.shape[0]
        ),
        "later_title_stable_updated_version_drift_median": float(
            later_title_stable_updated_drift[
                "jensen_shannon_distance"
            ].median()
        ),
        "later_title_stable_updated_version_drift_p90": float(
            later_title_stable_updated_drift[
                "jensen_shannon_distance"
            ].quantile(0.90)
        ),
        "matched_pair_share_above_later_title_stable_updated_drift_p90": float(
            (
                pairs["jensen_shannon_distance"]
                > later_title_stable_updated_drift[
                    "jensen_shannon_distance"
                ].quantile(0.90)
            ).mean()
        ),
        "matched_pair_median_contribution_jsd": float(pairs["jensen_shannon_distance"].median()),
        "version_drift_median_contribution_jsd": float(drift["jensen_shannon_distance"].median()),
        "caliper_sensitivity_rows": caliper_sensitivity.to_dict(orient="records"),
        "external_association_rows": external.to_dict(orient="records"),
    }
    (RESULTS / "two_layer_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
