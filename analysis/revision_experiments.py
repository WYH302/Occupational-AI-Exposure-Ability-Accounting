from __future__ import annotations

import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import sparse
from scipy.stats import norm, spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from analysis.build_manuscript_assets import (
    add_centrality_measures,
    build_measure_frame,
    build_occ_ability,
)
from analysis.build_validation_assets import load_census_soc_crosswalk
from analysis.revision_statistics import (
    aggregate_exposure_components,
    benjamini_hochberg,
    deflate_log_values,
    normalize_ability_names,
    robust_joint_pvalue,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RESULTS_DIR = ROOT / "results"
MASTER_DIR = RESULTS_DIR / "master"
TABLE_DIR = RESULTS_DIR / "tables"
FIGURE_DIR = RESULTS_DIR / "figures"
PAPER_TABLE_DIR = ROOT / "paper" / "tables"
PAPER_FIGURE_DIR = ROOT / "paper" / "figures"
RNG_SEED = 20260716
COMPONENTS = ["substitution_pressure", "relative_low_exposure_load", "reuse_intensity"]
COMPONENT_LABELS = {
    "substitution_pressure": "Positive-exposure load",
    "relative_low_exposure_load": "Relative low-exposure load",
    "reuse_intensity": "Reuse intensity",
}
# BLS CPI-U U.S. city average, all items, annual averages (1982--84=100).
# Source: https://www.bls.gov/cpi/tables/supplemental-files/historical-cpi-u-202412.pdf
CPI_U_ANNUAL = {
    2019: 255.657,
    2021: 270.970,
    2022: 292.655,
    2023: 304.702,
    2024: 313.689,
}


def format_pvalue(value: float) -> str:
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def ensure_directories() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def plot_primary_quartile_coefficients(
    unified: pd.DataFrame,
    dataset: str,
    output_paths: list[Path],
) -> list[Path]:
    required = {
        "Dataset",
        "AIOE quartile",
        "Component",
        "Coefficient",
        "Robust SE",
        "BH q-value",
    }
    missing = sorted(required.difference(unified.columns))
    if missing:
        raise KeyError(f"Coefficient plot columns missing: {missing}")

    quartiles = ["Q1", "Q2", "Q3", "Q4"]
    subset = unified.loc[unified["Dataset"].eq(dataset)].copy()
    expected_rows = len(quartiles) * len(COMPONENTS)
    if subset.shape[0] != expected_rows:
        raise ValueError(f"Expected {expected_rows} coefficient rows for {dataset}, found {subset.shape[0]}")

    colors = ["#0072B2", "#D55E00", "#009E73"]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.25), sharex=True, sharey=True)
    y_positions = np.arange(len(quartiles))
    for ax, component, color in zip(axes, COMPONENTS, colors, strict=True):
        label = COMPONENT_LABELS[component]
        component_rows = subset.loc[subset["Component"].eq(label)].copy()
        component_rows["AIOE quartile"] = pd.Categorical(
            component_rows["AIOE quartile"],
            categories=quartiles,
            ordered=True,
        )
        component_rows = component_rows.sort_values("AIOE quartile")
        if component_rows.shape[0] != len(quartiles):
            raise ValueError(f"Missing quartile rows for {label}")

        estimates = component_rows["Coefficient"].to_numpy(dtype=float)
        errors = 1.96 * component_rows["Robust SE"].to_numpy(dtype=float)
        qvalues = component_rows["BH q-value"].to_numpy(dtype=float)
        ax.axvline(0, color="#4D4D4D", linewidth=0.9, linestyle="--", zorder=1)
        ax.errorbar(
            estimates,
            y_positions,
            xerr=errors,
            fmt="none",
            ecolor=color,
            elinewidth=1.5,
            capsize=3,
            zorder=2,
        )
        for estimate, y_position, qvalue in zip(estimates, y_positions, qvalues, strict=True):
            ax.scatter(
                estimate,
                y_position,
                s=42,
                marker="o",
                facecolor=color if qvalue < 0.05 else "white",
                edgecolor=color,
                linewidth=1.4,
                zorder=3,
            )
        ax.set_title(label, fontsize=10.5, pad=8)
        ax.set_xlabel("Coefficient (95% CI)")
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(y_positions, quartiles)
    axes[0].set_ylabel("AIOE quartile")
    axes[0].invert_yaxis()
    fig.suptitle("Within-quartile associations with ACS annual earnings", fontsize=12, y=1.02)
    fig.text(
        0.5,
        -0.02,
        "Filled markers indicate BH-adjusted q < 0.05; intervals use robust standard errors.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()

    written: list[Path] = []
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        save_options = {"bbox_inches": "tight"}
        if path.suffix.lower() == ".png":
            save_options["dpi"] = 300
        fig.savefig(path, **save_options)
        written.append(path)
    plt.close(fig)
    return written


def save_table(frame: pd.DataFrame, stem: str, latex_decimals: int = 3) -> None:
    frame.to_csv(TABLE_DIR / f"{stem}.csv", index=False)
    latex = frame.copy()
    latex = latex.rename(columns={
        "Delta R2": r"$\Delta R^2$",
        "Partial R2": r"Partial $R^2$",
        "Partial R2 of interaction": r"Partial $R^2$ of interaction",
        "Out-of-fold R2": r"Out-of-fold $R^2$",
    })
    latex.columns = [
        str(column).replace("%", r"\%").replace("&", r"\&").replace("_", r"\_")
        for column in latex.columns
    ]
    for column in latex.columns:
        latex[column] = latex[column].map(
            lambda value: value.replace("%", r"\%").replace("&", r"\&")
            if isinstance(value, str)
            else value
        )
    text = latex.to_latex(
        index=False,
        escape=False,
        float_format=lambda value: f"{value:.{latex_decimals}f}",
    )
    (TABLE_DIR / f"{stem}.tex").write_text(text, encoding="utf-8")
    (PAPER_TABLE_DIR / f"{stem}.tex").write_text(text, encoding="utf-8")


def build_price_window_panels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_a_columns = [
        "Definition", "Q1 relative-load coefficient", "Q1 relative-load BH q-value",
        "Q3 reuse coefficient", "Q3 reuse BH q-value", "All-slope omnibus p-value",
    ]
    panel_b_columns = [
        "Definition", "Q1 joint p-value", "Q3 joint p-value",
        "Flexible-control joint p-value", "Flexible-control Delta R2", "Occupations",
    ]
    panel_a = frame.loc[:, panel_a_columns].rename(columns={
        "Q1 relative-load coefficient": "Q1 low-load coef.",
        "Q1 relative-load BH q-value": "Q1 BH $q$",
        "Q3 reuse coefficient": "Q3 reuse coef.",
        "Q3 reuse BH q-value": "Q3 BH $q$",
        "All-slope omnibus p-value": "All-slope $p$",
    })
    panel_b = frame.loc[:, panel_b_columns].rename(columns={
        "Q1 joint p-value": "Q1 joint $p$",
        "Q3 joint p-value": "Q3 joint $p$",
        "Flexible-control joint p-value": "Flexible joint $p$",
        "Flexible-control Delta R2": r"Flexible $\Delta R^2$",
    })
    latex_pvalue = lambda value: "$<0.001$" if value < 0.001 else f"{value:.3f}"
    for column in ["All-slope $p$"]:
        panel_a[column] = panel_a[column].map(latex_pvalue)
    for column in ["Q1 joint $p$", "Q3 joint $p$", "Flexible joint $p$"]:
        panel_b[column] = panel_b[column].map(latex_pvalue)
    return panel_a, panel_b


def build_onet_version_panels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel_a = frame.loc[:, [
        "O*NET release", "Occupations", "Abilities per occupation minimum",
        "Flexible-control low-load coefficient", "Low-load p-value",
        "Three-component joint p-value", "Delta R2",
    ]].rename(columns={
        "Abilities per occupation minimum": "Min. abilities",
        "Flexible-control low-load coefficient": "Low-load coefficient",
        "Low-load p-value": "Low-load $p$",
        "Three-component joint p-value": "Joint $p$",
        "Delta R2": r"$\Delta R^2$",
    })
    panel_b = frame.loc[:, [
        "O*NET release", "Cross-version Positive-exposure load rank rho",
        "Cross-version Relative low-exposure load rank rho",
        "Cross-version Reuse intensity rank rho",
    ]].rename(columns={
        "Cross-version Positive-exposure load rank rho": r"Positive-load rank $\rho$",
        "Cross-version Relative low-exposure load rank rho": r"Relative-load rank $\rho$",
        "Cross-version Reuse intensity rank rho": r"Reuse rank $\rho$",
    })
    return panel_a, panel_b


def zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    sd = values.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=series.index)
    return (values - values.mean()) / sd


def fit_model(formula: str, data: pd.DataFrame, weight_col: str | None = None):
    if weight_col is None:
        return smf.ols(formula, data=data).fit(cov_type="HC1")
    return smf.wls(formula, data=data, weights=data[weight_col]).fit(cov_type="HC1")


def fit_plain(formula: str, data: pd.DataFrame, weight_col: str | None = None):
    if weight_col is None:
        return smf.ols(formula, data=data).fit()
    return smf.wls(formula, data=data, weights=data[weight_col]).fit()


def coefficient(result: object, name: str) -> tuple[float, float, float]:
    names = list(result.model.exog_names)
    if name not in names:
        return np.nan, np.nan, np.nan
    index = names.index(name)
    return (
        float(np.asarray(result.params)[index]),
        float(np.asarray(result.bse)[index]),
        float(np.asarray(result.pvalues)[index]),
    )


def harmonize_measures(measures: pd.DataFrame) -> pd.DataFrame:
    renamed = measures.rename(
        columns={
            "substitute_pressure": "substitution_pressure",
            "augmentation_buffer": "relative_low_exposure_load",
            "hub_load": "reuse_intensity",
        }
    ).copy()
    for column in COMPONENTS:
        renamed[f"{column}_std"] = zscore(renamed[column])
    return renamed


def build_oe_change(panel: pd.DataFrame, measures: pd.DataFrame) -> pd.DataFrame:
    use = panel[panel["year"].isin([2021, 2022, 2023, 2024])].copy()
    use["period"] = np.where(use["year"] >= 2023, "post", "pre")
    change = use.groupby(["soc", "period"])["ln_wage"].mean().unstack()
    change["outcome_change"] = change["post"] - change["pre"]
    pre = panel[panel["year"] <= 2022].sort_values(["soc", "year"])
    trends = []
    for soc, group in pre.groupby("soc"):
        if group.shape[0] >= 2 and np.isfinite(group["ln_wage"]).all():
            slope = float(np.polyfit(group["year"], group["ln_wage"], 1)[0])
        else:
            slope = np.nan
        trends.append({"soc": soc, "pretrend_wage": slope})
    return (
        change[["outcome_change"]]
        .reset_index()
        .merge(measures, on="soc", how="inner", validate="one_to_one")
        .merge(pd.DataFrame(trends), on="soc", how="left", validate="one_to_one")
        .assign(analysis_weight=1.0)
    )


def build_acs_change(
    cells: pd.DataFrame,
    measures: pd.DataFrame,
    *,
    outcome_col: str = "ln_incwage",
    pre_years: tuple[int, ...] = (2019, 2021, 2022),
    post_years: tuple[int, ...] = (2023, 2024),
    equal_year_weights: bool = False,
) -> pd.DataFrame:
    selected_years = set(pre_years) | set(post_years)
    use = cells.loc[
        cells["YEAR"].isin(selected_years),
        ["soc", "YEAR", outcome_col, "weight_sum", "n"],
    ].copy()
    use["period"] = np.where(use["YEAR"].isin(post_years), "post", "pre")
    if equal_year_weights:
        collapsed = (
            use.groupby(["soc", "period"])
            .agg(
                weight=("weight_sum", "sum"),
                mean_outcome=(outcome_col, "mean"),
                records=("n", "sum"),
            )
            .reset_index()
        )
    else:
        use["weighted_outcome"] = use[outcome_col] * use["weight_sum"]
        collapsed = (
            use.groupby(["soc", "period"])
            .agg(
                weight=("weight_sum", "sum"),
                weighted_outcome=("weighted_outcome", "sum"),
                records=("n", "sum"),
            )
            .reset_index()
        )
        collapsed["mean_outcome"] = collapsed["weighted_outcome"] / collapsed["weight"]
    means = collapsed.pivot(index="soc", columns="period", values="mean_outcome")
    weights = collapsed.pivot(index="soc", columns="period", values="weight")
    change = means.assign(outcome_change=means["post"] - means["pre"])[["outcome_change"]]
    harmonic_weight = 2.0 / (1.0 / weights["pre"] + 1.0 / weights["post"])
    change["analysis_weight"] = harmonic_weight
    return change.reset_index().merge(measures, on="soc", how="inner", validate="one_to_one")


def build_acs_price_window_sensitivity(
    cells: pd.DataFrame,
    measures: pd.DataFrame,
    pretrends: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_cells = cells.copy()
    real_cells["ln_incwage_2024"] = deflate_log_values(
        real_cells["ln_incwage"],
        real_cells["YEAR"],
        CPI_U_ANNUAL,
        base_year=2024,
    )
    variants = [
        ("Nominal, person-year weighted", "ln_incwage", (2019, 2021, 2022), (2023, 2024), False, "analysis_weight"),
        ("2024 dollars, person-year weighted", "ln_incwage_2024", (2019, 2021, 2022), (2023, 2024), False, "analysis_weight"),
        ("2024 dollars, equal occupation weights", "ln_incwage_2024", (2019, 2021, 2022), (2023, 2024), False, None),
        ("Nominal, equal year weights", "ln_incwage", (2019, 2021, 2022), (2023, 2024), True, "analysis_weight"),
        ("2024 dollars, equal year weights", "ln_incwage_2024", (2019, 2021, 2022), (2023, 2024), True, "analysis_weight"),
        ("2024 dollars, 2021--22 vs 2023--24", "ln_incwage_2024", (2021, 2022), (2023, 2024), True, "analysis_weight"),
        ("Pandemic diagnostic: 2019 vs 2021--22", "ln_incwage_2024", (2019,), (2021, 2022), True, "analysis_weight"),
    ]
    rows = []
    registry_frames = []
    for label, outcome_col, pre_years, post_years, equal_year_weights, model_weight_col in variants:
        change = build_acs_change(
            real_cells,
            measures,
            outcome_col=outcome_col,
            pre_years=pre_years,
            post_years=post_years,
            equal_year_weights=equal_year_weights,
        ).merge(pretrends, on="soc", how="left", validate="one_to_one")
        unified, unified_registry = build_unified_quartile(change, label, model_weight_col)
        flexible, flexible_summary = build_flexible_controls(change, label, model_weight_col)
        q1_low = unified[
            (unified["AIOE quartile"] == "Q1")
            & (unified["Component"] == COMPONENT_LABELS["relative_low_exposure_load"])
        ].iloc[0]
        q3_reuse = unified[
            (unified["AIOE quartile"] == "Q3")
            & (unified["Component"] == COMPONENT_LABELS["reuse_intensity"])
        ].iloc[0]
        lookup = unified_registry.set_index("test")
        rows.append(
            {
                "Definition": label,
                "Q1 relative-load coefficient": q1_low["Coefficient"],
                "Q1 relative-load BH q-value": q1_low["BH q-value"],
                "Q3 reuse coefficient": q3_reuse["Coefficient"],
                "Q3 reuse BH q-value": q3_reuse["BH q-value"],
                "All-slope omnibus p-value": lookup.loc["all_12_slopes", "p_value"],
                "Q1 joint p-value": lookup.loc["Q1_three_components", "p_value"],
                "Q3 joint p-value": lookup.loc["Q3_three_components", "p_value"],
                "Flexible-control joint p-value": flexible_summary.iloc[0]["Joint robust Wald p-value"],
                "Flexible-control Delta R2": flexible_summary.iloc[0]["Delta R2"],
                "Occupations": change.shape[0],
            }
        )
        unified_registry = unified_registry.assign(
            family="acs_price_window_" + unified_registry["family"].astype(str),
            dataset=label,
        )
        registry_frames.append(unified_registry)
        flexible_q = benjamini_hochberg(flexible["p-value"])
        registry_frames.append(
            pd.DataFrame(
                {
                    "family": "acs_price_window_flexible_components",
                    "dataset": label,
                    "test": flexible["Component"],
                    "p_value": flexible["p-value"],
                    "adjusted_p_value": flexible_q,
                }
            )
        )
        registry_frames.append(
            pd.DataFrame(
                [
                    {
                        "family": "acs_price_window_flexible_joint",
                        "dataset": label,
                        "test": "three_components",
                        "p_value": flexible_summary.iloc[0]["Joint robust Wald p-value"],
                        "adjusted_p_value": np.nan,
                    }
                ]
            )
        )
    return pd.DataFrame(rows), pd.concat(registry_frames, ignore_index=True)


def add_broad_positive_exposure(measures: pd.DataFrame, occ_ability: pd.DataFrame) -> pd.DataFrame:
    broad = occ_ability.copy()
    broad["broad_positive_exposure"] = broad["w_norm"] * broad["ai_exposure"].clip(lower=0)
    broad = broad.groupby("soc", as_index=False)["broad_positive_exposure"].sum()
    output = measures.merge(broad, on="soc", how="left", validate="one_to_one")
    output["broad_positive_exposure_std"] = zscore(output["broad_positive_exposure"])
    return output


def build_component_anchors(occ_ability: pd.DataFrame) -> pd.DataFrame:
    ability = add_centrality_measures(occ_ability).groupby("Element Name", as_index=False).agg(
        ai_exposure=("ai_exposure", "mean"),
        reuse_weight=("hub_scale_strength", "mean"),
    )
    blocks = [
        (COMPONENT_LABELS["substitution_pressure"], ability.nlargest(5, "ai_exposure")),
        ("Relative low-exposure load", ability.nsmallest(5, "ai_exposure")),
        ("Reuse intensity", ability.nlargest(5, "reuse_weight")),
    ]
    rows = []
    for component, block in blocks:
        for _, row in block.iterrows():
            rows.append(
                {
                    "Component": component,
                    "Ability": row["Element Name"],
                    "AI exposure": float(row["ai_exposure"]),
                    "Reuse weight": float(row["reuse_weight"]),
                }
            )
    return pd.DataFrame(rows)


def build_ability_contribution_audit(
    occ_ability: pd.DataFrame,
    release: str,
) -> pd.DataFrame:
    detailed = add_centrality_measures(occ_ability).copy()
    weighted_reuse = detailed["w_norm"] * detailed["hub_scale_strength"]
    detailed["Substitution contribution"] = weighted_reuse * detailed["ai_exposure"].clip(lower=0)
    detailed["Relative low-exposure contribution"] = weighted_reuse * (-detailed["ai_exposure"]).clip(lower=0)
    detailed["Reuse contribution"] = weighted_reuse
    output = detailed.rename(
        columns={
            "Element Name": "Ability",
            "importance": "O*NET IM value",
            "w_norm": "Within-occupation weight",
            "ai_exposure": "FRS ability exposure",
            "hub_scale_strength": "Reuse weight",
        }
    )
    output.insert(0, "O*NET release", release.replace("_", "."))
    columns = [
        "O*NET release",
        "soc",
        "Ability",
        "O*NET IM value",
        "Within-occupation weight",
        "FRS ability exposure",
        "Reuse weight",
        "Substitution contribution",
        "Relative low-exposure contribution",
        "Reuse contribution",
    ]
    output = output[columns].sort_values(["soc", "Ability"]).reset_index(drop=True)
    counts = output.groupby("soc")["Ability"].nunique()
    if output.shape[0] != counts.shape[0] * 52 or not counts.eq(52).all():
        raise ValueError("Ability contribution audit must contain exactly 52 rows per occupation")
    return output


def build_measure_correlation_table(measures: pd.DataFrame) -> pd.DataFrame:
    columns = ["aioe", *COMPONENTS]
    labels = {"aioe": "AIOE", **{name: COMPONENT_LABELS[name] for name in COMPONENTS}}
    correlation = measures[columns].corr().rename(index=labels, columns=labels)
    correlation.index.name = "Measure"
    return correlation.reset_index()


def flexible_formula(component_terms: str = "") -> str:
    controls = (
        "bs(aioe, df=4, degree=3) + Q('Job Zone') + edu_score + "
        "pretrend_wage + C(major_group)"
    )
    return f"outcome_change ~ {controls}{component_terms}"


def top_group_jaccard(reference: pd.Series, candidate: pd.Series, share: float = 0.30) -> float:
    cutoff_reference = reference.quantile(1.0 - share)
    cutoff_candidate = candidate.quantile(1.0 - share)
    first = set(reference.index[reference >= cutoff_reference])
    second = set(candidate.index[candidate >= cutoff_candidate])
    return len(first & second) / len(first | second)


def fit_low_load_association(
    outcome: pd.DataFrame,
    variant: pd.DataFrame,
    weight_col: str | None,
) -> tuple[float, float]:
    sample = outcome.drop(columns=COMPONENTS + [f"{c}_std" for c in COMPONENTS], errors="ignore")
    sample = sample.merge(variant, on="soc", how="inner", validate="one_to_one")
    sample["low_variant_std"] = zscore(sample["relative_low_exposure_load"])
    required = [
        "outcome_change",
        "low_variant_std",
        "aioe",
        "Job Zone",
        "edu_score",
        "pretrend_wage",
        "major_group",
    ]
    sample = sample.dropna(subset=required)
    if sample["low_variant_std"].isna().all() or sample["low_variant_std"].std() == 0:
        return np.nan, np.nan
    result = fit_model(flexible_formula(" + low_variant_std"), sample, weight_col)
    coef, _, pvalue = coefficient(result, "low_variant_std")
    return coef, pvalue


def build_origin_sensitivity(
    release: str,
    oe_change: pd.DataFrame,
    acs_change: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    occ_ability, _, _, _, _ = build_occ_ability(release)
    occ_ability = add_centrality_measures(occ_ability)
    hub_col = "hub_scale_strength"
    baseline = aggregate_exposure_components(occ_ability, hub_col=hub_col).set_index("soc")
    variants: list[tuple[str, pd.DataFrame]] = []
    for shift in [-0.50, -0.25, 0.0, 0.25, 0.50]:
        variants.append(
            (
                f"Standardized exposure, origin shift {shift:+.2f} SD",
                aggregate_exposure_components(occ_ability, hub_col=hub_col, shift=shift),
            )
        )
    exposures = occ_ability.groupby("Element Name", as_index=False)["ai_exposure"].mean()
    for quantile in [0.25, 0.40, 0.50]:
        threshold = float(exposures["ai_exposure"].quantile(quantile))
        variants.append(
            (
                f"Exposure quantile threshold Q{int(quantile * 100)}",
                aggregate_exposure_components(occ_ability, hub_col=hub_col, threshold=threshold),
            )
        )
    rank_map = exposures.set_index("Element Name")["ai_exposure"].rank(pct=True) - 0.5
    ranked = occ_ability.copy()
    ranked["rank_exposure"] = ranked["Element Name"].map(rank_map)
    variants.append(
        (
            "Rank-centered exposure",
            aggregate_exposure_components(
                ranked,
                hub_col=hub_col,
                exposure_col="rank_exposure",
            ),
        )
    )
    appendix_d = pd.read_excel(RAW_DIR / "AIOE_DataAppendix.xlsx", sheet_name="Appendix D")
    appendix_d = appendix_d.rename(columns={appendix_d.columns[0]: "Element Name"})
    appendix_d = normalize_ability_names(appendix_d)
    raw_map = appendix_d.set_index("Element Name").mean(axis=1, numeric_only=True)
    uncentered = occ_ability.copy()
    uncentered["raw_application_mean"] = uncentered["Element Name"].map(raw_map)
    variants.append(
        (
            "Uncentered application-score mean",
            aggregate_exposure_components(
                uncentered,
                hub_col=hub_col,
                exposure_col="raw_application_mean",
            ),
        )
    )

    rows = []
    registry = []
    reference = baseline["relative_low_exposure_load"]
    for label, variant in variants:
        indexed = variant.set_index("soc")
        common = reference.index.intersection(indexed.index)
        low = indexed.loc[common, "relative_low_exposure_load"]
        rho = float(spearmanr(reference.loc[common], low).statistic) if low.nunique() > 1 else np.nan
        jaccard = top_group_jaccard(reference.loc[common], low) if low.nunique() > 1 else np.nan
        oe_coef, oe_p = fit_low_load_association(oe_change, variant, None)
        acs_coef, acs_p = fit_low_load_association(acs_change, variant, "analysis_weight")
        rows.append(
            {
                "Definition": label,
                "Rank rho": rho,
                "Top-30% Jaccard": jaccard,
                "OEWS coefficient": oe_coef,
                "OEWS p-value": oe_p,
                "ACS annual-earnings coefficient": acs_coef,
                "ACS p-value": acs_p,
                "Nonzero low-load occupations": int((low > 0).sum()),
            }
        )
        registry.extend(
            [
                {"family": "origin_sensitivity_exploratory", "dataset": "OEWS", "test": label, "p_value": oe_p},
                {"family": "origin_sensitivity_exploratory", "dataset": "ACS", "test": label, "p_value": acs_p},
            ]
        )

    loao_rows = []
    for ability in sorted(occ_ability["Element Name"].unique()):
        reduced = occ_ability[occ_ability["Element Name"] != ability].copy()
        reduced["w_norm"] = reduced["importance"] / reduced.groupby("soc")["importance"].transform("sum")
        variant = aggregate_exposure_components(reduced, hub_col=hub_col)
        indexed = variant.set_index("soc")
        common = reference.index.intersection(indexed.index)
        low = indexed.loc[common, "relative_low_exposure_load"]
        oe_coef, oe_p = fit_low_load_association(oe_change, variant, None)
        acs_coef, acs_p = fit_low_load_association(acs_change, variant, "analysis_weight")
        loao_rows.append(
            {
                "Omitted ability": ability,
                "Rank rho": float(spearmanr(reference.loc[common], low).statistic),
                "Top-30% Jaccard": top_group_jaccard(reference.loc[common], low),
                "OEWS coefficient": oe_coef,
                "OEWS p-value": oe_p,
                "ACS annual-earnings coefficient": acs_coef,
                "ACS p-value": acs_p,
            }
        )
    loao = pd.DataFrame(loao_rows)
    for _, row in loao.iterrows():
        registry.extend(
            [
                {
                    "family": "leave_one_ability_out_52",
                    "dataset": "OEWS",
                    "test": row["Omitted ability"],
                    "p_value": row["OEWS p-value"],
                },
                {
                    "family": "leave_one_ability_out_52",
                    "dataset": "ACS annual earnings",
                    "test": row["Omitted ability"],
                    "p_value": row["ACS p-value"],
                },
            ]
        )
    registry_frame = pd.DataFrame(registry)
    registry_frame["adjusted_p_value"] = np.nan
    for _, indexes in registry_frame.groupby(["family", "dataset"]).groups.items():
        finite = registry_frame.loc[indexes, "p_value"].notna()
        selected = registry_frame.loc[indexes].index[finite]
        if len(selected):
            registry_frame.loc[selected, "adjusted_p_value"] = benjamini_hochberg(
                registry_frame.loc[selected, "p_value"]
            )
    return pd.DataFrame(rows), loao, registry_frame


def summarize_loao(loao: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, coef_col, p_col in [
        ("OEWS published estimates", "OEWS coefficient", "OEWS p-value"),
        ("ACS annual earnings", "ACS annual-earnings coefficient", "ACS p-value"),
    ]:
        rows.append(
            {
                "Dataset": dataset,
                "Coefficient minimum": loao[coef_col].min(),
                "Coefficient median": loao[coef_col].median(),
                "Coefficient maximum": loao[coef_col].max(),
                "Positive-sign share": float((loao[coef_col] > 0).mean()),
                "Share with p below 0.05": float((loao[p_col] < 0.05).mean()),
                "Minimum rank rho": loao["Rank rho"].min(),
                "Minimum top-30% Jaccard": loao["Top-30% Jaccard"].min(),
            }
        )
    return pd.DataFrame(rows)


def build_unified_quartile(
    sample: pd.DataFrame,
    dataset: str,
    weight_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = sample.dropna(subset=["outcome_change", "aioe", *COMPONENTS]).copy()
    data["quartile"] = pd.qcut(data["aioe"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    terms = []
    for quartile in ["Q1", "Q2", "Q3", "Q4"]:
        mask = data["quartile"] == quartile
        for component in COMPONENTS:
            local = zscore(data.loc[mask, component])
            name = f"{component}_{quartile}"
            data[name] = 0.0
            data.loc[mask, name] = local
            terms.append(name)
    formula = "outcome_change ~ C(quartile) + " + " + ".join(terms)
    result = fit_model(formula, data, weight_col)
    rows = []
    pvalues = []
    for quartile in ["Q1", "Q2", "Q3", "Q4"]:
        for component in COMPONENTS:
            name = f"{component}_{quartile}"
            coef, se, pvalue = coefficient(result, name)
            pvalues.append(pvalue)
            rows.append(
                {
                    "Dataset": dataset,
                    "AIOE quartile": quartile,
                    "Component": COMPONENT_LABELS[component],
                    "Coefficient": coef,
                    "Robust SE": se,
                    "p-value": pvalue,
                }
            )
    adjusted = benjamini_hochberg(pvalues)
    for row, qvalue in zip(rows, adjusted, strict=True):
        row["BH q-value"] = qvalue

    registry = [
        {
            "family": "unified_quartile_12_slopes",
            "dataset": dataset,
            "test": term,
            "p_value": pvalue,
            "adjusted_p_value": qvalue,
        }
        for term, pvalue, qvalue in zip(terms, pvalues, adjusted, strict=True)
    ]
    registry.append(
        {
            "family": "unified_quartile_omnibus",
            "dataset": dataset,
            "test": "all_12_slopes",
            "p_value": robust_joint_pvalue(result, terms),
            "adjusted_p_value": np.nan,
        }
    )
    for quartile in ["Q1", "Q2", "Q3", "Q4"]:
        names = [f"{component}_{quartile}" for component in COMPONENTS]
        registry.append(
            {
                "family": "unified_quartile_omnibus",
                "dataset": dataset,
                "test": f"{quartile}_three_components",
                "p_value": robust_joint_pvalue(result, names),
                "adjusted_p_value": np.nan,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(registry)


def build_flexible_controls(
    sample: pd.DataFrame,
    dataset: str,
    weight_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [
        "outcome_change",
        "aioe",
        "Job Zone",
        "edu_score",
        "pretrend_wage",
        "major_group",
        *[f"{component}_std" for component in COMPONENTS],
    ]
    data = sample.dropna(subset=required).copy()
    base_formula = flexible_formula()
    component_names = [f"{component}_std" for component in COMPONENTS]
    full_formula = flexible_formula(" + " + " + ".join(component_names))
    base_plain = fit_plain(base_formula, data, weight_col)
    full_plain = fit_plain(full_formula, data, weight_col)
    robust = fit_model(full_formula, data, weight_col)
    rows = []
    for component, name in zip(COMPONENTS, component_names, strict=True):
        coef, se, pvalue = coefficient(robust, name)
        rows.append(
            {
                "Dataset": dataset,
                "Component": COMPONENT_LABELS[component],
                "Coefficient": coef,
                "Robust SE": se,
                "p-value": pvalue,
            }
        )
    partial_r2 = (full_plain.rsquared - base_plain.rsquared) / max(1e-12, 1.0 - base_plain.rsquared)
    summary = pd.DataFrame(
        [
            {
                "Dataset": dataset,
                "Specification": "AIOE cubic spline + structure + pretrend",
                "Joint robust Wald p-value": robust_joint_pvalue(robust, component_names),
                "Delta R2": full_plain.rsquared - base_plain.rsquared,
                "Partial R2": partial_r2,
                "N occupations": data.shape[0],
            }
        ]
    )
    return pd.DataFrame(rows), summary


def build_theory_interaction(
    sample: pd.DataFrame,
    dataset: str,
    weight_col: str | None,
) -> pd.DataFrame:
    data = sample.copy()
    for name in ["broad_positive_exposure", "reuse_intensity"]:
        data[f"{name}_centered"] = zscore(data[name])
    data["exposure_reuse_interaction"] = (
        data["broad_positive_exposure_centered"] * data["reuse_intensity_centered"]
    )
    required = [
        "outcome_change",
        "broad_positive_exposure_centered",
        "reuse_intensity_centered",
        "exposure_reuse_interaction",
        "substitution_pressure_std",
        "relative_low_exposure_load_std",
        "aioe",
        "Job Zone",
        "edu_score",
        "pretrend_wage",
        "major_group",
    ]
    data = data.dropna(subset=required)
    base = flexible_formula(
        " + substitution_pressure_std + relative_low_exposure_load_std + "
        "broad_positive_exposure_centered + reuse_intensity_centered"
    )
    full = base + " + exposure_reuse_interaction"
    reduced_plain = fit_plain(base, data, weight_col)
    full_plain = fit_plain(full, data, weight_col)
    robust = fit_model(full, data, weight_col)
    coef, se, pvalue = coefficient(robust, "exposure_reuse_interaction")
    design_columns = [
        "substitution_pressure_std",
        "relative_low_exposure_load_std",
        "broad_positive_exposure_centered",
        "reuse_intensity_centered",
        "exposure_reuse_interaction",
    ]
    design = data[design_columns].apply(zscore).dropna().to_numpy()
    partial_r2 = (full_plain.rsquared - reduced_plain.rsquared) / max(1e-12, 1.0 - reduced_plain.rsquared)
    return pd.DataFrame(
        [
            {
                "Dataset": dataset,
                "Interaction coefficient": coef,
                "Robust SE": se,
                "p-value": pvalue,
                "Partial R2 of interaction": partial_r2,
                "Standardized condition number": float(np.linalg.cond(design)),
                "N occupations": data.shape[0],
            }
        ]
    )


def prediction_pipeline(include_components: bool) -> Pipeline:
    numeric = ["aioe", "Job Zone", "edu_score", "pretrend_wage"]
    if include_components:
        numeric += COMPONENTS
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
            ("numeric", StandardScaler(), [column for column in numeric if column != "aioe"]),
            ("major", OneHotEncoder(handle_unknown="ignore"), ["major_group"]),
        ]
    )
    return Pipeline([("features", transformer), ("model", LinearRegression())])


def cross_validate_increment(
    sample: pd.DataFrame,
    dataset: str,
    weight_col: str | None,
) -> pd.DataFrame:
    required = [
        "outcome_change",
        "aioe",
        "Job Zone",
        "edu_score",
        "pretrend_wage",
        "major_group",
        *COMPONENTS,
    ]
    data = sample.dropna(subset=required).reset_index(drop=True)
    schemes = [
        ("5-fold grouped by 2-digit SOC", GroupKFold(n_splits=5).split(data, groups=data["major_group"])),
        ("10-fold random", KFold(n_splits=10, shuffle=True, random_state=RNG_SEED).split(data)),
    ]
    rows = []
    for scheme, split_iterator in schemes:
        splits = list(split_iterator)
        predictions: dict[str, np.ndarray] = {
            "AIOE spline + structure": np.full(data.shape[0], np.nan),
            "Baseline + three components": np.full(data.shape[0], np.nan),
        }
        for include_components, label in [
            (False, "AIOE spline + structure"),
            (True, "Baseline + three components"),
        ]:
            for train, test in splits:
                model = prediction_pipeline(include_components)
                fit_kwargs = {}
                if weight_col is not None:
                    fit_kwargs["model__sample_weight"] = data.loc[train, weight_col]
                model.fit(data.loc[train], data.loc[train, "outcome_change"], **fit_kwargs)
                predictions[label][test] = model.predict(data.loc[test])
            y = data["outcome_change"].to_numpy()
            pred = predictions[label]
            rows.append(
                {
                    "Dataset": dataset,
                    "Validation scheme": scheme,
                    "Model": label,
                    "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
                    "MAE": float(mean_absolute_error(y, pred)),
                    "Out-of-fold R2": float(r2_score(y, pred)),
                    "N occupations": data.shape[0],
                }
            )
    return pd.DataFrame(rows)


def cluster_meat(scores: sparse.csr_matrix, groups: pd.Series) -> np.ndarray:
    codes, uniques = pd.factorize(groups, sort=True)
    indicator = sparse.csr_matrix(
        (np.ones(len(codes)), (codes, np.arange(len(codes)))),
        shape=(len(uniques), len(codes)),
    )
    sums = indicator @ scores
    meat = (sums.T @ sums).toarray()
    n, k = scores.shape
    g = len(uniques)
    correction = (g / max(1, g - 1)) * ((n - 1) / max(1, n - k))
    return correction * meat


def state_year_two_way_model(
    cells: pd.DataFrame,
    measures: pd.DataFrame,
    *,
    weighted: bool,
    min_cell_n: int,
) -> tuple[pd.DataFrame, float]:
    measure_cols = ["soc", *[f"{component}_std" for component in COMPONENTS]]
    data = cells.drop(columns=[column for column in cells if column.endswith("_std")], errors="ignore")
    data = data.merge(measures[measure_cols], on="soc", how="inner", validate="many_to_one")
    data = data[data["n"] >= min_cell_n].dropna(subset=["ln_incwage", "weight_sum", *measure_cols[1:]]).copy()
    for component in COMPONENTS:
        data[f"{component}_post"] = data[f"{component}_std"] * (data["YEAR"] >= 2023).astype(float)
    interaction = sparse.csr_matrix(data[[f"{component}_post" for component in COMPONENTS]].to_numpy())
    occ_codes = pd.Categorical(data["soc"]).codes
    occ = sparse.csr_matrix(
        (np.ones(data.shape[0]), (np.arange(data.shape[0]), occ_codes)),
        shape=(data.shape[0], int(occ_codes.max()) + 1),
    )
    state_year_label = data["STATEFIP"].astype(str) + "_" + data["YEAR"].astype(str)
    sy_codes = pd.Categorical(state_year_label).codes
    state_year = sparse.csr_matrix(
        (np.ones(data.shape[0]), (np.arange(data.shape[0]), sy_codes)),
        shape=(data.shape[0], int(sy_codes.max()) + 1),
    )[:, 1:]
    design = sparse.hstack([interaction, occ, state_year], format="csr")
    weights = np.sqrt(data["weight_sum"].to_numpy() if weighted else np.ones(data.shape[0]))
    weighted_design = design.multiply(weights[:, None]).tocsr()
    weighted_y = data["ln_incwage"].to_numpy() * weights
    beta = sparse.linalg.lsqr(weighted_design, weighted_y, atol=1e-11, btol=1e-11)[0]
    residual = weighted_y - weighted_design @ beta
    scores = weighted_design.multiply(residual[:, None]).tocsr()
    intersection = data["STATEFIP"].astype(str) + "_" + data["soc"].astype(str)
    meat = (
        cluster_meat(scores, data["soc"])
        + cluster_meat(scores, data["STATEFIP"])
        - cluster_meat(scores, intersection)
    )
    bread = np.linalg.pinv((weighted_design.T @ weighted_design).toarray())
    covariance = bread @ meat @ bread
    rows = []
    for index, component in enumerate(COMPONENTS):
        se = float(np.sqrt(max(0.0, covariance[index, index])))
        pvalue = float(2.0 * norm.sf(abs(beta[index] / se))) if se > 0 else np.nan
        rows.append(
            {
                "Weighting": "Person-weight WLS" if weighted else "Unweighted cells",
                "Minimum cell n": min_cell_n,
                "Component": COMPONENT_LABELS[component],
                "Coefficient": float(beta[index]),
                "Two-way cluster SE": se,
                "p-value": pvalue,
                "N cells": data.shape[0],
                "Occupations": data["soc"].nunique(),
                "States": data["STATEFIP"].nunique(),
            }
        )
    beta3 = beta[:3]
    cov3 = covariance[:3, :3]
    joint = float(beta3 @ np.linalg.pinv(cov3) @ beta3)
    joint_p = float(1.0 - __import__("scipy").stats.chi2.cdf(joint, 3))
    return pd.DataFrame(rows), joint_p


def build_state_year_robustness(cells: pd.DataFrame, measures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    registry = []
    for weighted, min_n in [(True, 1), (False, 1), (True, 30)]:
        table, joint_p = state_year_two_way_model(cells, measures, weighted=weighted, min_cell_n=min_n)
        rows.append(table)
        registry.append(
            {
                "family": "state_year_joint_exploratory",
                "dataset": "ACS state x occupation x year",
                "test": f"joint_weighted_{weighted}_min_n_{min_n}",
                "p_value": joint_p,
                "adjusted_p_value": np.nan,
            }
        )
    combined = pd.concat(rows, ignore_index=True)
    combined["BH q-value"] = benjamini_hochberg(combined["p-value"])
    for _, row in combined.iterrows():
        registry.append(
            {
                "family": "state_year_nine_components_exploratory",
                "dataset": "ACS state x occupation x year",
                "test": (
                    f"{row['Weighting']}; min_n={int(row['Minimum cell n'])}; "
                    f"{row['Component']}"
                ),
                "p_value": row["p-value"],
                "adjusted_p_value": row["BH q-value"],
            }
        )
    return combined, pd.DataFrame(registry)


def valid_onet_archive(release: str) -> bool:
    path = RAW_DIR / f"db_{release}_excel.zip"
    if not path.exists():
        return False
    try:
        with ZipFile(path) as archive:
            return archive.testzip() is None
    except BadZipFile:
        return False


def onet_release_available(release: str) -> bool:
    extracted = RAW_DIR / f"db_{release}_excel"
    required = [extracted / "Abilities.xlsx", extracted / "Job Zones.xlsx"]
    education_candidates = [
        extracted / "Education, Training, and Experience.xlsx",
        extracted / "Education.xlsx",
    ]
    extracted_available = (
        all(path.exists() and path.stat().st_size > 0 for path in required)
        and any(path.exists() and path.stat().st_size > 0 for path in education_candidates)
    )
    return extracted_available or valid_onet_archive(release)


def build_version_sensitivity(
    acs_cells: pd.DataFrame,
    panel: pd.DataFrame,
    primary_release: str,
) -> pd.DataFrame:
    price_adjusted_cells = acs_cells.copy()
    price_adjusted_cells["ln_incwage_2024"] = deflate_log_values(
        price_adjusted_cells["ln_incwage"],
        price_adjusted_cells["YEAR"],
        CPI_U_ANNUAL,
        base_year=2024,
    )
    releases = [primary_release]
    for release in ["27_1", "30_2", "30_3"]:
        if release not in releases and onet_release_available(release):
            releases.append(release)
    measure_frames = {}
    rows = []
    for release in releases:
        measure = harmonize_measures(build_measure_frame(release))
        occ_ability, _, _, _, _ = build_occ_ability(release)
        measure = add_broad_positive_exposure(measure, occ_ability)
        measure_frames[release] = measure
        oe = build_oe_change(panel, measure)
        acs = build_acs_change(
            price_adjusted_cells,
            measure,
            outcome_col="ln_incwage_2024",
        ).merge(
            oe[["soc", "pretrend_wage"]], on="soc", how="left", validate="one_to_one"
        )
        flexible_rows, flexible_summary = build_flexible_controls(
            acs, f"ACS annual earnings, O*NET {release.replace('_', '.')}", "analysis_weight"
        )
        low = flexible_rows[flexible_rows["Component"] == COMPONENT_LABELS["relative_low_exposure_load"]].iloc[0]
        rows.append(
            {
                "O*NET release": f"Release {release.replace('_', '.')}",
                "Occupations": measure.shape[0],
                "Abilities per occupation minimum": int(measure["ability_count"].min()),
                "Flexible-control low-load coefficient": low["Coefficient"],
                "Low-load p-value": low["p-value"],
                "Three-component joint p-value": flexible_summary.iloc[0]["Joint robust Wald p-value"],
                "Delta R2": flexible_summary.iloc[0]["Delta R2"],
            }
        )
    reference = primary_release
    for release, row in zip(releases, rows, strict=True):
        common = measure_frames[reference].merge(
            measure_frames[release],
            on="soc",
            suffixes=(f"_{reference}", f"_{release}"),
        )
        for component in COMPONENTS:
            if release == reference:
                rho = 1.0
            else:
                rho = spearmanr(
                    common[f"{component}_{reference}"],
                    common[f"{component}_{release}"],
                ).statistic
            row[f"Cross-version {COMPONENT_LABELS[component]} rank rho"] = float(rho)
    return pd.DataFrame(rows)


def build_crosswalk_audit(crosswalk: pd.DataFrame, measures: pd.DataFrame, acs_cells: pd.DataFrame) -> pd.DataFrame:
    mapped = crosswalk.merge(measures[["soc"]], on="soc", how="inner")
    return pd.DataFrame(
        [
            {
                "Audit item": "Exact four-digit Census occupation rows",
                "Value": f"{crosswalk.shape[0]:,}",
                "Interpretation": "Range/header rows excluded before parsing",
            },
            {
                "Audit item": "Unique Census occupation codes",
                "Value": f"{crosswalk['census_occ'].nunique():,}",
                "Interpretation": "One exact crosswalk row per Census code",
            },
            {
                "Audit item": "Crosswalk rows matching measurement universe",
                "Value": f"{mapped.shape[0]:,}",
                "Interpretation": "Aggregate SOC rows without a detailed measure drop here",
            },
            {
                "Audit item": "Final ACS occupations",
                "Value": f"{acs_cells['soc'].nunique():,}",
                "Interpretation": "Positive annual wage and salary income, ages 18--64",
            },
            {
                "Audit item": "Final unweighted ACS records",
                "Value": f"{int(acs_cells.groupby(['soc', 'YEAR'])['n'].first().sum()):,}",
                "Interpretation": "Derived from final cached occupation-year cells",
            },
            {
                "Audit item": "Final ACS person-weight sum",
                "Value": f"{int(round(acs_cells.groupby(['soc', 'YEAR'])['weight_sum'].first().sum())):,}",
                "Interpretation": "Person weights, not a count of observed records",
            },
        ]
    )


def build_primary_quartile_summary(
    unified: pd.DataFrame,
    registry: pd.DataFrame,
    dataset: str,
) -> pd.DataFrame:
    subset = unified[unified["Dataset"] == dataset]
    rows = []
    for quartile in ["Q1", "Q2", "Q3", "Q4"]:
        row = {"AIOE quartile": quartile}
        for component in COMPONENTS:
            match = subset[
                (subset["AIOE quartile"] == quartile)
                & (subset["Component"] == COMPONENT_LABELS[component])
            ].iloc[0]
            coefficient = float(match["Coefficient"])
            robust_se = float(match["Robust SE"])
            lower = coefficient - 1.96 * robust_se
            upper = coefficient + 1.96 * robust_se
            row[COMPONENT_LABELS[component]] = (
                f"{coefficient:.3f} [{lower:.3f}, {upper:.3f}]; "
                f"q={float(match['BH q-value']):.4f}"
            )
        omnibus = registry[
            (registry["dataset"] == dataset)
            & (registry["test"] == f"{quartile}_three_components")
        ].iloc[0]
        row["Joint p-value"] = format_pvalue(float(omnibus["p_value"]))
        rows.append(row)
    return pd.DataFrame(rows)


def build_flexible_main_table(
    components: pd.DataFrame,
    summary: pd.DataFrame,
    predictive: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for dataset in summary["Dataset"]:
        row = {"Dataset": dataset}
        stress = summary[summary["Dataset"] == dataset].iloc[0]
        row["Joint p-value"] = format_pvalue(float(stress["Joint robust Wald p-value"]))
        row[r"$\Delta R^2$"] = f"{float(stress['Delta R2']):.3f}"
        grouped = predictive[
            (predictive["Dataset"] == dataset)
            & (predictive["Validation scheme"] == "5-fold grouped by 2-digit SOC")
        ].set_index("Model")
        row["Change in RMSE, components minus baseline"] = f"{float(
            grouped.loc["Baseline + three components", "RMSE"]
            - grouped.loc["AIOE spline + structure", "RMSE"]
        ):+.4f}"
        row["Occupations"] = f"{int(stress['N occupations']):,}"
        rows.append(row)
    return pd.DataFrame(rows)


def build_state_main_table(state_year: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    subset = state_year[
        (state_year["Weighting"] == "Person-weight WLS")
        & (state_year["Minimum cell n"] == 1)
    ]
    rows = []
    for component in COMPONENTS:
        match = subset[subset["Component"] == COMPONENT_LABELS[component]].iloc[0]
        rows.append(
            {
                "Component": COMPONENT_LABELS[component],
                "Estimate (SE)": (
                    f"{match['Coefficient']:.3f} ({match['Two-way cluster SE']:.3f})"
                ),
                "$p$": format_pvalue(float(match["p-value"])),
                "BH $q$": f"{match['BH q-value']:.4f}",
            }
        )
    joint = registry[registry["test"] == "joint_weighted_True_min_n_1"].iloc[0]
    rows.append(
        {
            "Component": "Joint Wald test",
            "Estimate (SE)": "--",
            "$p$": format_pvalue(float(joint["p_value"])),
            "BH $q$": "--",
        }
    )
    return pd.DataFrame(rows)


def build_additional_inference_registry(
    flexible_components: pd.DataFrame,
    flexible_summary: pd.DataFrame,
    theory: pd.DataFrame,
    version: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for dataset, group in flexible_components.groupby("Dataset", sort=False):
        qvalues = benjamini_hochberg(group["p-value"])
        for (_, record), qvalue in zip(group.iterrows(), qvalues, strict=True):
            rows.append(
                {
                    "family": "flexible_three_components_exploratory",
                    "dataset": dataset,
                    "test": record["Component"],
                    "p_value": record["p-value"],
                    "adjusted_p_value": qvalue,
                }
            )
        joint = flexible_summary[flexible_summary["Dataset"] == dataset].iloc[0]
        rows.append(
            {
                "family": "flexible_three_component_joint",
                "dataset": dataset,
                "test": "three_components",
                "p_value": joint["Joint robust Wald p-value"],
                "adjusted_p_value": np.nan,
            }
        )
    theory_qvalues = benjamini_hochberg(theory["p-value"])
    for (_, record), qvalue in zip(theory.iterrows(), theory_qvalues, strict=True):
        rows.append(
            {
                "family": "direct_exposure_reuse_interaction",
                "dataset": record["Dataset"],
                "test": "broad_positive_exposure_x_reuse",
                "p_value": record["p-value"],
                "adjusted_p_value": qvalue,
            }
        )
    version_tests = []
    for _, record in version.iterrows():
        version_tests.extend(
            [
                (record["O*NET release"], "relative_low_exposure_load", record["Low-load p-value"]),
                (record["O*NET release"], "three_component_joint", record["Three-component joint p-value"]),
            ]
        )
    version_qvalues = benjamini_hochberg([item[2] for item in version_tests])
    for (release, test, pvalue), qvalue in zip(version_tests, version_qvalues, strict=True):
        rows.append(
            {
                "family": "onet_version_sensitivity_exploratory",
                "dataset": f"ACS annual earnings; {release}",
                "test": test,
                "p_value": pvalue,
                "adjusted_p_value": qvalue,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ensure_directories()
    primary_release = "27_1" if onet_release_available("27_1") else "30_3"
    measures = harmonize_measures(build_measure_frame(primary_release))
    occ_ability, _, _, _, _ = build_occ_ability(primary_release)
    measures = add_broad_positive_exposure(measures, occ_ability)
    panel = pd.read_csv(RESULTS_DIR / "panel_annual.csv", dtype={"soc": str})
    acs_cells = pd.read_csv(MASTER_DIR / "acs_occupation_year_cells.csv", dtype={"soc": str})
    state_cells = pd.read_csv(MASTER_DIR / "acs_state_occupation_year_cells.csv", dtype={"soc": str})
    acs_cells["ln_incwage_2024"] = deflate_log_values(
        acs_cells["ln_incwage"], acs_cells["YEAR"], CPI_U_ANNUAL, base_year=2024
    )
    oe_change = build_oe_change(panel, measures)
    pretrends = oe_change[["soc", "pretrend_wage"]]
    acs_change = build_acs_change(
        acs_cells,
        measures,
        outcome_col="ln_incwage_2024",
    ).merge(
        pretrends, on="soc", how="left", validate="one_to_one"
    )

    origin, loao, origin_registry = build_origin_sensitivity(primary_release, oe_change, acs_change)
    unified_oe, registry_oe = build_unified_quartile(oe_change, "OEWS published estimates", None)
    unified_acs, registry_acs = build_unified_quartile(
        acs_change, "ACS annual earnings", "analysis_weight"
    )
    flexible_oe, flexible_summary_oe = build_flexible_controls(
        oe_change, "OEWS published estimates", None
    )
    flexible_acs, flexible_summary_acs = build_flexible_controls(
        acs_change, "ACS annual earnings", "analysis_weight"
    )
    theory = pd.concat(
        [
            build_theory_interaction(oe_change, "OEWS published estimates", None),
            build_theory_interaction(acs_change, "ACS annual earnings", "analysis_weight"),
        ],
        ignore_index=True,
    )
    predictive = pd.concat(
        [
            cross_validate_increment(oe_change, "OEWS published estimates", None),
            cross_validate_increment(acs_change, "ACS annual earnings", "analysis_weight"),
        ],
        ignore_index=True,
    )
    state_year, state_registry = build_state_year_robustness(state_cells, measures)
    crosswalk = load_census_soc_crosswalk()
    crosswalk_audit = build_crosswalk_audit(crosswalk, measures, acs_cells)
    version = build_version_sensitivity(acs_cells, panel, primary_release)
    price_window, price_window_registry = build_acs_price_window_sensitivity(
        acs_cells,
        measures,
        pretrends,
    )
    ability_audit = build_ability_contribution_audit(occ_ability, primary_release)
    ability_audit.to_csv(RESULTS_DIR / "ability_contribution_audit.csv", index=False)

    save_table(origin, "table_origin_sensitivity")
    save_table(build_component_anchors(occ_ability), "table_component_anchors")
    save_table(build_measure_correlation_table(measures), "table_measure_correlations")
    save_table(loao, "table_leave_one_ability_out")
    save_table(summarize_loao(loao), "table_leave_one_ability_out_summary")
    unified_all = pd.concat([unified_oe, unified_acs], ignore_index=True)
    save_table(unified_all, "table_unified_quartile")
    plot_primary_quartile_coefficients(
        unified_all,
        dataset="ACS annual earnings",
        output_paths=[
            FIGURE_DIR / "within_quartile_coefficients.pdf",
            FIGURE_DIR / "within_quartile_coefficients.png",
            PAPER_FIGURE_DIR / "within_quartile_coefficients.pdf",
            PAPER_FIGURE_DIR / "within_quartile_coefficients.png",
        ],
    )
    save_table(pd.concat([flexible_oe, flexible_acs], ignore_index=True), "table_flexible_controls_components")
    save_table(
        pd.concat([flexible_summary_oe, flexible_summary_acs], ignore_index=True),
        "table_flexible_controls_summary",
    )
    save_table(theory, "table_theory_interaction")
    save_table(predictive, "table_predictive_validation")
    save_table(state_year, "table_state_year_two_way_cluster", latex_decimals=4)
    save_table(crosswalk_audit, "table_crosswalk_audit")
    save_table(version, "table_onet_version_sensitivity")
    save_table(price_window, "table_acs_price_window_sensitivity")
    version_panel_a, version_panel_b = build_onet_version_panels(version)
    price_panel_a, price_panel_b = build_price_window_panels(price_window)
    save_table(version_panel_a, "table_onet_version_sensitivity_panel_a")
    save_table(version_panel_b, "table_onet_version_sensitivity_panel_b")
    save_table(price_panel_a, "table_acs_price_window_panel_a")
    save_table(price_panel_b, "table_acs_price_window_panel_b")
    additional_registry = build_additional_inference_registry(
        pd.concat([flexible_oe, flexible_acs], ignore_index=True),
        pd.concat([flexible_summary_oe, flexible_summary_acs], ignore_index=True),
        theory,
        version,
    )
    registry = pd.concat(
        [
            origin_registry,
            registry_oe,
            registry_acs,
            state_registry,
            price_window_registry,
            additional_registry,
        ],
        ignore_index=True,
    )
    registry = registry[["family", "dataset", "test", "p_value", "adjusted_p_value"]]
    registry.to_csv(RESULTS_DIR / "statistical_test_registry.csv", index=False)
    save_table(
        build_primary_quartile_summary(
            unified_all,
            registry,
            "ACS annual earnings",
        ),
        "table_primary_quartile_summary",
    )
    save_table(
        build_flexible_main_table(
            pd.concat([flexible_oe, flexible_acs], ignore_index=True),
            pd.concat([flexible_summary_oe, flexible_summary_acs], ignore_index=True),
            predictive,
        ),
        "table_flexible_main",
    )
    save_table(build_state_main_table(state_year, registry), "table_state_year_main")
    metrics = {
        "primary_onet_release": primary_release.replace("_", "."),
        "primary_onet_release_reason": (
            "Pre-ChatGPT November 2022 snapshot" if primary_release == "27_1" else "Historical archive unavailable"
        ),
        "ability_count": int(measures["ability_count"].min()),
        "oe_occupations": int(oe_change.shape[0]),
        "acs_occupations": int(acs_change.shape[0]),
        "acs_final_unweighted_records": int(acs_cells["n"].sum()),
        "acs_hourly_wage_available": False,
        "acs_hourly_wage_blocker": "Local extract lacks usual weekly hours and weeks worked",
        "acs_price_basis": "2024 dollars using BLS CPI-U annual averages",
        "ability_contribution_rows": int(ability_audit.shape[0]),
    }
    (RESULTS_DIR / "revision_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
