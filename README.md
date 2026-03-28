# Analysis Code for: Magnetoencephalography biomarkers for assessing myelin content and neuronal function in acute optic neuritis

Analysis scripts accompanying the paper submitted to [Brain Communications](https://academic.oup.com/braincomms) (Oxford University Press).

## Scripts

| Script | Paper Result | Description |
|--------|-------------|-------------|
| `correlations.py` | Fig 4 | Pearson correlations between MEG/neurological metrics and ophthalmological outcomes, with Shapiro-Wilk normality tests and scatter plots |
| `regression_models.py` | Table 3 | PCA visual impairment index + OLS regression models predicting the index from structural (GCL) and functional (P100/M100) variables |
| `icc_reproducibility.py` | Table 2, Supp Table 1 | Intraclass correlation coefficients for P100 latency test-retest reproducibility across runs, directions, raters, and modalities |
| `harmonic_count_comparison.py` | Fig 3C | Paired t-tests comparing harmonic count and cumulated SNR between affected and fellow eyes, with boxplots |

## Input Data

Each script reads from a CSV file (path defined at the top of each script):

- `correlations.py` and `regression_models.py` read `data.csv`
- `icc_reproducibility.py` reads `data_test_retest.csv`
- `harmonic_count_comparison.py` reads `data_harmonics.csv`

### Required CSV Columns

**data.csv**: `patient_number`, `nb_h`, `oct`, `on_length`, `peak_stc`, `peak_ophtalmo`, `sloan`, `latency_variance`, `hue_err_x`, `etdrs_score`

Note: `ophtalmo_pca_1` and `ophtalmo_pca_2` are computed at runtime by `regression_models.py` via PCA.

**data_test_retest.csv**: `patient_number`, `peak`, `run`, `direction`, `rater`, `type`, `eye`, `angle`, `color`, `is_patient`, `is_eye_affected`, `are_meaned_runs`, `ignore_test_retest`

Note: `item_info` is computed internally by `icc_reproducibility.py`.

**data_harmonics.csv**: `patient_code`, `area`, `eye`, `complete_eye_type`, `index_followup`, `nb_h`, `cumulated_a`

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python correlations.py
python regression_models.py
python icc_reproducibility.py
python harmonic_count_comparison.py
```

Output figures are saved to `./figures/`.
