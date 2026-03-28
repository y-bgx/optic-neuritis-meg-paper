"""
ICC test-retest reproducibility (Table 2).

Computes intraclass correlation coefficients (ICC) for P100 latency
measurements across runs, directions, raters, and modalities (VEP vs MEG).
Produces scatter plots and exports results to Excel/pickle.

Source: test_retest.py
"""
import functools
import logging
import operator
import os
from math import sqrt
from pathlib import Path
from typing import Dict, Generator

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg

CSV_PATH = Path(__file__).parent / "data_test_retest.csv"


def df_friendly_concat(
    df: pd.DataFrame, columns: list, behavior_if_missing: str = "ignore"
) -> pd.Series:
    """Concatenate string representations of multiple columns into a single key."""

    def iterator_column(
        df: pd.DataFrame, columns: list
    ) -> Generator[pd.Series, None, None]:
        for c in columns:
            try:
                yield df[c].astype(str)
            except KeyError:
                if behavior_if_missing == "ignore":
                    continue
                else:
                    raise

    return functools.reduce(operator.add, iterator_column(df, columns))


def compact_name(filter: Dict, raters: str) -> str:
    """Build a compact descriptive name from filter and raters parameters."""
    name_elements = {"comp": raters, **filter}
    return "_".join(f"{k}_{v}" for k, v in name_elements.items())


def icc(df: pd.DataFrame, raters: str, filter: Dict) -> pd.DataFrame:
    """Compute ICC for a given comparison (raters) with data filtered by filter.

    Also produces a scatter plot of the two measurement occasions and computes
    test-retest variability (TRV) and delta statistics.
    """
    df2 = df
    df2 = df2[df2["angle"] == "60"]
    df2 = df2[df2["rater"] != "YC"]
    df2 = df2[df2["direction"] != "mean_right_left"]
    if "are_meaned_runs" not in filter:
        df2 = df2[~df2["ignore_test_retest"]]
        df2 = df2.dropna(subset=["run"], inplace=False)
        df2 = df2[df2["run"] <= 2]
    for k, v in filter.items():
        df2 = df2[df2[k] == v]
    cols = ["eye", "direction", "angle", "patient_number", "rater", "type", "run"]
    cols.remove(raters)
    df2["item_info"] = df_friendly_concat(df2, cols)
    unique_patient_numbers = [str(i) for i in df2["patient_number"].unique()]
    patient_numbers = (
        str(len(unique_patient_numbers)) + " : " + ",".join(unique_patient_numbers)
    )
    try:
        icc_df = pg.intraclass_corr(
            data=df2,
            raters=raters,
            targets="item_info",
            ratings="peak",
            nan_policy="omit",
        )
    except Exception:
        logging.exception(f"icc failed for {raters}, {filter}")
        return pd.DataFrame()

    try:
        df_pivot = df2.pivot_table(index="item_info", columns=raters, values="peak")
    except Exception:
        logging.exception(f"pivot_table failed for {raters}, {filter}")
        return pd.DataFrame()

    rater_values = df2[raters].unique()
    if len(rater_values) > 2:
        try:
            rater_values = rater_values[~np.isnan(rater_values)]
        except Exception:
            pass
    if len(rater_values) != 2:
        raise ValueError(
            f"Expected exactly 2 rater values, got {len(rater_values)}: {rater_values}"
        )
    df_pivot["delta"] = df_pivot[rater_values[0]] - df_pivot[rater_values[1]]
    icc_df["delta_standard_deviation"] = df_pivot["delta"].std()
    icc_df["patient_numbers"] = patient_numbers

    # Test-retest variability (TRV)
    df_pivot["delta2"] = df_pivot["delta"] ** 2
    sw_square = df_pivot["delta2"].mean() / 2
    trv = 1.96 * sqrt(sw_square)

    df_pivot["delta"] = df_pivot["delta"].abs()
    mn = df_pivot["delta"].mean()
    icc_df["delta"] = mn
    icc_df["trv"] = trv

    name = compact_name(filter, raters)

    # --- Scatter plot ---
    labels_colors = {"k": "healthy", "red": "affected", "green": "fellow"}
    f, ax = plt.subplots()
    labels = None
    for color in ["k", "red", "green"]:
        df2_color = df2[df2["color"] == color]
        runs = df2_color.pivot_table(index="item_info", columns=raters, values="peak")
        runs = runs.dropna(inplace=False)
        if runs.empty:
            continue
        columns = runs.columns
        run1 = list(runs[columns[0]])
        run2 = list(runs[columns[1]])
        ax.scatter(
            run1,
            run2,
            c=color,
            s=10,
            alpha=0.5,
            label=labels_colors[color],
        )
        ax.tick_params(axis="both", which="major", labelsize="large")
        labels = columns[0], columns[1]
        ax.legend()
    if labels is None:
        labels = "1 ?", "2 ?"
    if tuple(labels) == (1, 2):
        labels = "1", "2"
    plt.xlabel(f"{raters} {labels[0]}", fontsize="large")
    plt.ylabel(f"{raters} {labels[1]}", fontsize="large")
    plt.suptitle("peaks")
    ax.legend(loc="upper left", fontsize="large")
    ax.plot([85, 180], [85, 180], color="grey", linestyle="dashed")

    f.savefig(f"./figures/{name}.png")
    f.savefig(f"./figures/{name}.svg")
    plt.close()
    return icc_df


def clone_with_augmented_filter(d: Dict, filter: Dict) -> Dict:
    """Copy a config dict while merging additional filter keys."""
    return {**d, "filter": {**d["filter"], **filter}}


def iterator_all_icc_to_perform(iccs_to_perform: list) -> Generator[Dict, None, None]:
    """Yield all ICC configurations, expanding each base config into
    sub-analyses (affected/fellow eye, patient/healthy splits)."""
    for d in iccs_to_perform:
        obtention_type = d["filter"].get("type")
        if obtention_type == "ophtalmo":
            yield d
            yield clone_with_augmented_filter(
                d, {"is_patient": True, "is_eye_affected": True}
            )
            yield clone_with_augmented_filter(
                d, {"is_patient": True, "is_eye_affected": False}
            )
        elif obtention_type == "stc":
            yield d
            yield clone_with_augmented_filter(d, {"is_patient": False})
            yield clone_with_augmented_filter(
                d, {"is_patient": True, "is_eye_affected": True}
            )
            yield clone_with_augmented_filter(
                d, {"is_patient": True, "is_eye_affected": False}
            )
        elif d["raters"] == "type":
            yield clone_with_augmented_filter(
                d, {"is_patient": True, "are_meaned_runs": True}
            )
            yield clone_with_augmented_filter(
                d,
                {"is_patient": True, "is_eye_affected": False, "are_meaned_runs": True},
            )
            yield clone_with_augmented_filter(
                d,
                {"is_patient": True, "is_eye_affected": True, "are_meaned_runs": True},
            )


def compute_iccs_and_create_figures(largest_df: pd.DataFrame) -> None:
    """Run all ICC analyses and save results to Excel and pickle."""
    os.makedirs("./figures", exist_ok=True)
    iccs_to_perform = [
        dict(raters="run", filter={"rater": "Ysoline", "type": "ophtalmo"}),
        dict(raters="run", filter={"rater": "Ysoline", "type": "stc"}),
        dict(raters="run", filter={"rater": "Celine", "type": "ophtalmo"}),
        dict(raters="run", filter={"rater": "Celine", "type": "stc"}),
        dict(raters="direction", filter={"rater": "Ysoline", "type": "stc"}),
        dict(raters="direction", filter={"rater": "Ysoline", "type": "ophtalmo"}),
        dict(raters="direction", filter={"rater": "Celine", "type": "ophtalmo"}),
        dict(raters="direction", filter={"rater": "Celine", "type": "stc"}),
        dict(raters="rater", filter={"type": "ophtalmo"}),
        dict(raters="rater", filter={"type": "stc"}),
        dict(raters="type", filter={"is_patient": True, "rater": "Ysoline"}),
    ]
    all_iccs_performed = []
    keys = []
    for icc_info in iterator_all_icc_to_perform(iccs_to_perform):
        all_iccs_performed.append(icc(largest_df, **icc_info))
        keys.append(compact_name(**icc_info))
    non_empty = [(k, df) for k, df in zip(keys, all_iccs_performed) if not df.empty]
    if non_empty:
        result_keys, result_dfs = zip(*non_empty)
        results = pd.concat(result_dfs, keys=result_keys)
        results.to_excel("./results_icc.xlsx")
        results.to_pickle("./results_icc.pkl")
    else:
        logging.warning("All ICC computations returned empty results")


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    compute_iccs_and_create_figures(df)
