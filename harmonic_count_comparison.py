"""
Paired t-test + boxplot for harmonic count comparison (Fig 3C).

Compares harmonic count and cumulated SNR between affected and fellow eyes
using paired t-tests (with Shapiro-Wilk normality checks on the differences).
Produces boxplots per brain area and metric, with significance bars and
descriptive statistics exported as JSON sidecar files.

Source: harmonics_analysis/main.py  year_zero_multiple_areas_analysis()
"""
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import seaborn as sns

CSV_PATH = Path(__file__).parent / "data_harmonics.csv"


def significancy_shortlabel(p_val: float) -> str:
    """Return significance stars for a p-value."""
    if p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    return "ns"


def sidecar_txt_stat_infos(filepath: str, metadata: dict) -> None:
    """Write statistical metadata as a JSON sidecar file alongside a figure."""
    _, extension = os.path.splitext(filepath)
    sidecar_filepath = filepath.replace(extension, ".json")
    with open(sidecar_filepath, "w") as f:
        f.write(json.dumps(metadata, indent=4, sort_keys=True))


def add_significance_bars(
    asteriks: str, max_val: float, positions: tuple = (0, 1)
) -> None:
    """Add a significance bar with stars above a boxplot."""
    x1, x2 = positions
    y, h, col = max_val + 3, 2, "k"
    plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col)
    plt.text((x1 + x2) * 0.5, y + h, asteriks, ha="center", va="bottom", color=col)


def get_tests_info_for_plot(
    affected: pd.DataFrame,
    df: pd.DataFrame,
    fellow: pd.DataFrame,
    matched: bool,
    metric_studied: str,
) -> tuple:
    """Run paired or independent t-test with normality checks.

    Returns (significance_label, max_metric_value, p_value).
    """
    max_metric = df[metric_studied].max()

    if matched:
        # Paired t-test: test normality of differences
        merged = fellow[["patient_code", metric_studied]].merge(
            affected[["patient_code", metric_studied]],
            on="patient_code",
            suffixes=("_fellow", "_affected"),
        )
        differences = (
            merged[f"{metric_studied}_fellow"] - merged[f"{metric_studied}_affected"]
        )

        if len(differences) >= 3:
            shapiro_result = scipy.stats.shapiro(differences)
            print("\n=== NORMALITY TEST FOR PAIRED T-TEST ===")
            print(f"Metric: {metric_studied}")
            print("Testing normality of differences (Fellow - Affected)")
            print(f"Shapiro-Wilk statistic: {shapiro_result.statistic:.6f}")
            print(f"Shapiro-Wilk p-value: {shapiro_result.pvalue:.6f}")
            if shapiro_result.pvalue < 0.05:
                print("WARNING: Differences are NOT normally distributed (p < 0.05)")
                print(
                    "Consider using Wilcoxon signed-rank test instead of paired t-test"
                )
            else:
                print("Differences appear normally distributed (p >= 0.05)")
            print(f"Sample size: {len(differences)}")
            print("=" * 40 + "\n")

        vals = scipy.stats.ttest_rel(
            fellow[metric_studied],
            affected[metric_studied],
        )
    else:
        # Independent t-test: test normality of each group
        if len(fellow[metric_studied]) >= 3:
            shapiro_fellow = scipy.stats.shapiro(fellow[metric_studied])
            print("\n=== NORMALITY TEST FOR INDEPENDENT T-TEST ===")
            print(f"Metric: {metric_studied}")
            print(
                f"Fellow eye - Shapiro-Wilk statistic: {shapiro_fellow.statistic:.6f}"
            )
            print(f"Fellow eye - Shapiro-Wilk p-value: {shapiro_fellow.pvalue:.6f}")
            if shapiro_fellow.pvalue < 0.05:
                print("WARNING: Fellow eye is NOT normally distributed (p < 0.05)")
            else:
                print("Fellow eye appears normally distributed (p >= 0.05)")

        if len(affected[metric_studied]) >= 3:
            shapiro_affected = scipy.stats.shapiro(affected[metric_studied])
            print(
                "Affected eye - Shapiro-Wilk statistic:"
                f" {shapiro_affected.statistic:.6f}"
            )
            print(f"Affected eye - Shapiro-Wilk p-value: {shapiro_affected.pvalue:.6f}")
            if shapiro_affected.pvalue < 0.05:
                print("WARNING: Affected eye is NOT normally distributed (p < 0.05)")
            else:
                print("Affected eye appears normally distributed (p >= 0.05)")
            print(
                f"Sample sizes: Fellow={len(fellow[metric_studied])},"
                f" Affected={len(affected[metric_studied])}"
            )
            print("=" * 40 + "\n")

        vals = scipy.stats.ttest_ind(
            fellow[metric_studied],
            affected[metric_studied],
        )
    print(vals)
    p_val = vals.pvalue
    asteriks = significancy_shortlabel(p_val)
    return asteriks, max_metric, p_val


def year_zero_multiple_areas_analysis(df_original: pd.DataFrame) -> None:
    """Compare harmonic count and cumulated SNR between affected and fellow
    eyes across brain areas, using paired t-tests and boxplots."""
    os.makedirs("./figures", exist_ok=True)
    nice_names = {"cumulated_a": "Cumulated SNR", "nb_h": "Harmonic Count"}

    for idx in [0, 1, 10]:
        print(f"\n{'=' * 60}")
        print(f"Follow-up index: {idx}")
        print(f"{'=' * 60}")

        df = df_original[
            (df_original["index_followup"] == idx) & (df_original["area"] != "all")
        ]
        for area_selected in df["area"].unique():
            df_area = df[df["area"] == area_selected]
            print(f"\n--- Area: {area_selected} ---")

            for metric_studied in {"nb_h", "cumulated_a"}:
                print(f"\nMetric: {metric_studied}")
                df_area_no_duplicates = df_area[
                    [metric_studied, "patient_code", "complete_eye_type", "eye"]
                ].drop_duplicates()
                df_area_no_duplicates = df_area_no_duplicates.dropna()
                fellow = df_area_no_duplicates[
                    df_area_no_duplicates["eye"] == "fellow eye"
                ]
                affected = df_area_no_duplicates[
                    df_area_no_duplicates["eye"] == "affected eye"
                ]
                if set(fellow["patient_code"]) != set(affected["patient_code"]):
                    raise ValueError(
                        "Mismatched patients between fellow and affected eyes: "
                        f"{set(fellow['patient_code']).symmetric_difference(set(affected['patient_code']))}"
                    )

                # --- Boxplot ---
                df_boxplot = df_area[
                    [metric_studied, "patient_code", "eye", "complete_eye_type"]
                ]
                df_boxplot["eye"] = df_boxplot["eye"].apply(lambda x: x.capitalize())
                df_boxplot = df_boxplot.dropna()
                df_boxplot.drop_duplicates(inplace=True)
                sns.boxplot(x="eye", y=metric_studied, data=df_boxplot)
                sns.swarmplot(x="eye", y=metric_studied, data=df_boxplot, color=".25")
                plt.ylabel(nice_names.get(metric_studied, metric_studied))

                # --- Paired t-test ---
                asteriks, max_metric, p_val = get_tests_info_for_plot(
                    affected, df, fellow, True, metric_studied
                )
                add_significance_bars(asteriks, max_metric * 1, (0, 1))

                plt.title(
                    f"area: {area_selected}, metric: {metric_studied}, p_val: {p_val}"
                )
                filetitle = (
                    f"./figures/gcarea_{area_selected}_{metric_studied}_idx_{idx}.png"
                )
                plt.savefig(filetitle)
                plt.close()

                # --- Descriptive statistics ---
                fellow_median = fellow[metric_studied].median()
                fellow_iqr = scipy.stats.iqr(fellow[metric_studied])
                fellow_sem = scipy.stats.sem(fellow[metric_studied])
                affected_median = affected[metric_studied].median()
                affected_iqr = scipy.stats.iqr(affected[metric_studied])
                affected_sem = scipy.stats.sem(affected[metric_studied])

                metadata = {
                    "area": area_selected,
                    "metric": metric_studied,
                    "p_val": p_val,
                    "fellow_median": fellow_median,
                    "fellow_iqr": fellow_iqr,
                    "fellow_sem": fellow_sem,
                    "affected_median": affected_median,
                    "affected_iqr": affected_iqr,
                    "affected_sem": affected_sem,
                    "idx": idx,
                }
                sidecar_txt_stat_infos(filetitle, metadata)


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    year_zero_multiple_areas_analysis(df)
