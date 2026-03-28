"""
PCA visual index + OLS regressions (Table 3).

Constructs a composite visual index via PCA on three ophthalmological measures
(hue error, ETDRS score, low-contrast VA), then fits OLS regression models
predicting this visual index from structural (GCL volume) and functional
(P100 latency) variables.

Source: df_merger.py  add_pca_analysis() + regression_analysis()
"""
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA

CSV_PATH = Path(__file__).parent / "data.csv"


def add_pca_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Build a PCA visual index from hue_err_x, etdrs_score, sloan.

    Adds columns ophtalmo_pca_1, ophtalmo_pca_2 to the dataframe.
    """
    selected_columns = ["hue_err_x", "etdrs_score", "sloan"]
    df_pca = df[selected_columns + ["patient_number"]].drop_duplicates()
    df_pca = df_pca.drop(columns=["patient_number"])
    df_pca = df_pca.dropna()

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_pca)
    principal_df = pd.DataFrame(
        data=principal_components, columns=["ophtalmo_pca_1", "ophtalmo_pca_2"]
    )
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    df = df.reset_index(drop=True)
    principal_df = principal_df.reset_index(drop=True)
    df = pd.concat([principal_df, df], axis=1)
    return df


def fix_alternated_nan_peaks(df_no_dup: pd.DataFrame) -> pd.DataFrame:
    """Resolve rows where peak_ophtalmo and peak_stc have NaNs on
    alternating lines for the same patient (due to data layout)."""
    df_patient_number_ophtalmo_peak = df_no_dup[["patient_number", "peak_ophtalmo"]]
    df_patient_number_stc_peak = df_no_dup[["patient_number", "peak_stc"]]
    df_without_peaks = df_no_dup.drop(columns=["peak_ophtalmo", "peak_stc"])
    df_final = df_without_peaks.merge(df_patient_number_stc_peak, on="patient_number")
    df_final = (
        df_final.merge(df_patient_number_ophtalmo_peak, on="patient_number")
        .dropna()
        .drop_duplicates()
    )
    df_final["nb_h"] = df_final["nb_h"].astype(float)
    return df_final


def regression_analysis(df: pd.DataFrame) -> None:
    """Fit OLS models predicting the PCA visual index from
    structural (GCL) and functional (P100 latency) predictors."""
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.copy()
    df_no_dup = df[
        ["ophtalmo_pca_1", "nb_h", "patient_number", "oct", "peak_stc", "peak_ophtalmo"]
    ].drop_duplicates()
    df_final = fix_alternated_nan_peaks(df_no_dup)

    formulas = [
        "ophtalmo_pca_1 ~ oct + peak_ophtalmo",
        "ophtalmo_pca_1 ~ oct + peak_stc",
        "ophtalmo_pca_1 ~ oct + peak_stc + peak_ophtalmo + nb_h",
    ]
    for formula in formulas:
        print(f"\n{'#' * 50}")
        print(f"# {formula}")
        print(f"{'#' * 50}")
        results = smf.ols(formula, data=df_final).fit()
        print(results.summary())
        print(f"P-values:\n{results.pvalues}\n")


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    df = add_pca_analysis(df)
    regression_analysis(df)
