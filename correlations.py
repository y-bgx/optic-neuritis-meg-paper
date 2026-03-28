"""
Pearson correlations + normality tests + scatter plots (Fig 4).

Computes pairwise Pearson correlations between MEG/neurological metrics
and ophthalmological outcomes, with Shapiro-Wilk
normality tests before each correlation. Produces a correlation matrix and
individual scatter plots saved to ./figures/.

Source: df_merger.py  correlation_studies()
"""
import itertools as it
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import statsmodels.api as sm

AREA = "V1"
CSV_PATH = Path(__file__).parent / "data.csv"


def correlation_studies(dataframe2: pd.DataFrame) -> None:
    """Run all pairwise Pearson correlations between neurological and
    ophthalmological metrics, with normality checks and scatter plots."""
    os.makedirs("./figures", exist_ok=True)

    neurological_metrics = [
        "nb_h",
        "oct",
        "on_length",
        "peak_stc",
        "peak_ophtalmo",
    ]
    ophtalmological_outcomes = [
        "sloan",
        "oct",
        "peak_stc",
        "peak_ophtalmo",
        "latency_variance",
    ]

    nice_names = {
        "on_length": "ON length",
        "nb_h": "Harmonic count",
        "cumulated_a": "Cumulated Amplitude",
        "peak_stc": "P100 Latency (MEG)",
        "peak_ophtalmo": "P100 Latency (VEP)",
        "sloan": "Low contrast 2.5% VA",
        "oct": "Macular GCL volume (mm³)",
        "oct_rnflm": "mRNFLM thickness (µm)",
        "oct_rnflt": "Temporal RNFLM thickness (µm)",
        "localizer_objects": "Localizer Amplitude (Objects)",
        "localizer_faces": "Localizer Amplitude (Faces)",
    }
    all_metrics = list(set(neurological_metrics + ophtalmological_outcomes))
    dataframe_for_corr = dataframe2[all_metrics]
    dataframe_for_corr = dataframe_for_corr.loc[
        :, ~dataframe_for_corr.columns.duplicated()
    ].copy()

    # --- Correlation matrix ---
    corr = dataframe_for_corr.corr()
    sm.graphics.plot_corr(
        corr, xnames=[nice_names.get(i, i) for i in dataframe_for_corr.columns]
    )
    plt.savefig(f"./figures/{AREA}_fig_correlation_matrix.png")
    plt.savefig(f"./figures/{AREA}_fig_correlation_matrix.svg")

    # --- Pairwise correlations ---
    for neurological_metric, ophtalmological_outcome in it.product(
        neurological_metrics, ophtalmological_outcomes
    ):
        if neurological_metric == ophtalmological_outcome:
            continue
        if (
            neurological_metric == "oct"
            and "sloan" not in ophtalmological_outcome
            and "latency_variance" not in ophtalmological_outcome
        ):
            continue
        dataframe3 = dataframe2[
            [neurological_metric, ophtalmological_outcome, "patient_number"]
        ]
        dataframe3.drop_duplicates(inplace=True)

        left = (
            dataframe3[[ophtalmological_outcome, "patient_number"]]
            .drop_duplicates()
            .dropna()
        )
        right = (
            dataframe3[[neurological_metric, "patient_number"]]
            .drop_duplicates()
            .dropna()
        )
        patients_in_left = set(left["patient_number"].unique())
        patients_in_right = set(right["patient_number"].unique())
        patients_in_full_df_after_dropping_nas = set(
            dataframe3.dropna()["patient_number"].unique()
        )
        patients_in_left_and_right_but_not_in_full_df = patients_in_left.intersection(
            patients_in_right
        ).difference(patients_in_full_df_after_dropping_nas)
        if patients_in_left_and_right_but_not_in_full_df:
            comparison_set = {"peak_ophtalmo", "peak_stc"}
            if comparison_set == {neurological_metric, ophtalmological_outcome}:
                dataframe3 = left.merge(right, on="patient_number", how="inner")
            else:
                raise ValueError(
                    f"patients {patients_in_left_and_right_but_not_in_full_df}"
                    " are in left and right but not in full df"
                )

        dataframe3 = dataframe3[[neurological_metric, ophtalmological_outcome]]
        dataframe3 = dataframe3.dropna()
        dataframe3 = dataframe3.loc[:, ~dataframe3.columns.duplicated()].copy()
        print(
            f"Correlation between {neurological_metric} and {ophtalmological_outcome}"
        )
        try:
            # Shapiro-Wilk normality test before computing correlation
            if len(dataframe3) >= 3:
                print("\n=== NORMALITY TEST FOR PEARSON CORRELATION ===")
                print(
                    f"Testing correlation between {neurological_metric}"
                    f" and {ophtalmological_outcome}"
                )

                if len(dataframe3[neurological_metric].dropna()) >= 3:
                    shapiro_neuro = scipy.stats.shapiro(
                        dataframe3[neurological_metric].dropna()
                    )
                    print(f"{neurological_metric}:")
                    print(f"  Shapiro-Wilk statistic: {shapiro_neuro.statistic:.6f}")
                    print(f"  Shapiro-Wilk p-value: {shapiro_neuro.pvalue:.6f}")
                    if shapiro_neuro.pvalue < 0.05:
                        print(
                            f"  WARNING: {neurological_metric}"
                            " is NOT normally distributed (p < 0.05)"
                        )
                        print(
                            "  Consider using Spearman correlation instead of Pearson"
                        )
                    else:
                        print(
                            f"  {neurological_metric}"
                            " appears normally distributed (p >= 0.05)"
                        )

                if len(dataframe3[ophtalmological_outcome].dropna()) >= 3:
                    shapiro_ophtalmo = scipy.stats.shapiro(
                        dataframe3[ophtalmological_outcome].dropna()
                    )
                    print(f"{ophtalmological_outcome}:")
                    print(f"  Shapiro-Wilk statistic: {shapiro_ophtalmo.statistic:.6f}")
                    print(f"  Shapiro-Wilk p-value: {shapiro_ophtalmo.pvalue:.6f}")
                    if shapiro_ophtalmo.pvalue < 0.05:
                        print(
                            f"  WARNING: {ophtalmological_outcome}"
                            " is NOT normally distributed (p < 0.05)"
                        )
                        print(
                            "  Consider using Spearman correlation instead of Pearson"
                        )
                    else:
                        print(
                            f"  {ophtalmological_outcome}"
                            " appears normally distributed (p >= 0.05)"
                        )

                print(f"Sample size: {len(dataframe3)}")
                print("=" * 50 + "\n")

            vals2 = scipy.stats.pearsonr(
                dataframe3[neurological_metric], dataframe3[ophtalmological_outcome]
            )
        except ValueError:
            print("ValueError")
            continue
        if ophtalmological_outcome == "oct":
            plot = sns.lmplot(
                x=ophtalmological_outcome, y=neurological_metric, data=dataframe3
            )
            if neurological_metric == "nb_h":
                plt.yticks(np.arange(0, 20, 2.0))
            plt.ylabel(
                nice_names.get(neurological_metric, neurological_metric), size=15
            )
            plt.xlabel(
                nice_names.get(ophtalmological_outcome, ophtalmological_outcome),
                size=15,
            )
        else:
            plot = sns.lmplot(
                y=ophtalmological_outcome, x=neurological_metric, data=dataframe3
            )
            if neurological_metric == "nb_h":
                plt.xticks(np.arange(0, 20, 2.0))
            if ophtalmological_outcome == "sloan":
                plot.set(ylim=(-2, 50))
            plt.xlabel(
                nice_names.get(neurological_metric, neurological_metric), size=15
            )
            plt.ylabel(
                nice_names.get(ophtalmological_outcome, ophtalmological_outcome),
                size=15,
            )
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(str({"pval": round(vals2[1], 4), "r": round(vals2[0], 4)}))
        plt.savefig(
            f"./figures/{AREA}correlation_{neurological_metric}_{ophtalmological_outcome}.png",
            bbox_inches="tight",
        )
        plt.savefig(
            f"./figures/{AREA}correlation_{neurological_metric}_{ophtalmological_outcome}.svg",
            bbox_inches="tight",
        )
        plt.close()

    plt.close("all")


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    correlation_studies(df)
