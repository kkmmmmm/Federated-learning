# Federated learning for 30-day mortality in acute myocardial infarction (GUSTO-I)

Reproducible code for the study comparing **local**, **centralized** and
**federated** logistic-regression models predicting 30-day mortality after acute
myocardial infarction, using the GUSTO-I trial data (40,830 patients across 16
regions). Federated learning is implemented with the
**[Flower](https://flower.ai) framework** (FedAvg).

Three paradigms are compared, each under four regularisation strategies
(none / L1 / L2 / elastic-net):

* **Local** — trained on a single region.
* **Centralized** — trained on pooled patient-level data.
* **Federated (FL)** — FedAvg over per-region clients, sharing only model
  parameters (never raw data).

---

## Data

`GUSTO_sterberg.csv` — columns: `DAY30` (30-day mortality outcome), `REGL`
(region 1–16), and the 17 predictors `SEX, AGE, KILLIP, DIA, HYP, HRT, ANT, PMI,
HEI, WEI, SMK, HTN, LIP, PAN, FAM, STE, TTR`. Place it one directory above the
package, or set the `GUSTO_CSV` environment variable.

The GUSTO-I dataset is publicly available from the Duke Clinical Research
Institute and can be loaded in R via
`load(url("http://hbiostat.org/data/repo/gusto.rda"))`.

---

## Install & run

```bash
pip install -r requirements.txt
python run_all.py                 # full run (1000 bootstrap resamples)
python run_all.py --boot 100      # faster
python run_all.py --penalties l1  # subset of regularisation strategies
python regen_figures.py           # rebuild figures/Excel from saved results
```

Outputs are written to `outputs/`:

| File | Contents |
|------|----------|
| `results.xlsx` | Table 2 / S3 (within-region mean±SD AUROC → **Figure 1**), raw folds, between-region long format, FL convergence + recalibration, coefficients |
| `AUROC.xlsx` | **Figure 2** / S2–S4 source data (between-region AUROC), per-region block layout |
| `calibration_slope.xlsx` | **Figure 3** / S5–S7 source data |
| `calibration_intercept.xlsx` | **Figure 4** / S8–S10 source data (includes the uncorrected FedAvg series) |
| `PCA.xlsx` | **Figure 5** source data |
| `figures/*.png`, `*.pdf` | Publication-format figures, named by manuscript figure number |

In the publication-ready figure-data workbooks and figures, **FL** denotes the
*recalibrated* federated model (FedAvg followed by the federated intercept
correction), matching the manuscript. The uncorrected FedAvg model is shown only
as an extra series on the calibration-intercept output (Figure 4 / S8–S10) and as
a `FedAvg` row in the `coefficients` sheet. The comprehensive `results.xlsx`
workbook additionally contains raw diagnostic sheets (`within_region_folds`,
`between_region`, `PCA_*`) in which the internal labels `FL` (uncorrected FedAvg)
and `FL_recal` (recalibrated) are both retained.

---

## Methods

### Standardisation (federated)
Predictors are standardised on one common scale built from per-region aggregate
statistics only — each client shares `count / sum / sum-of-squares`, and the
server reconstructs the exact pooled mean/SD. No patient-level data are shared,
and coefficient averaging happens on a single common scale
(`src/data_utils.py:federated_scaler`).

### Federated averaging (FedAvg)
Each communication round, every region (client) warm-starts from the current
global model, takes a local update on its own standardised data — using a
regularisation strength `C` (and elastic-net `l1_ratio`) selected by
cross-validation **within that client** — and returns its regression
coefficients; the server aggregates them by sample-size-weighted averaging. All
16 regions participate in every round. Each FedAvg model is trained for a fixed
60 communication rounds, and the global training log-loss is monitored to
characterise convergence rather than used as a stopping rule (`src/flower_fl.py`).

### Federated intercept recalibration
Because FedAvg averages per-region coefficients, the aggregated model does not in
general satisfy the pooled intercept score equation, leaving a systematic
calibration-in-the-large offset under variable-selecting penalties (L1 /
elastic-net). After the fixed 60 rounds the global model is therefore finalised
with a single **intercept-recalibration step**: the aggregated slopes are held fixed and
the intercept is shifted so that, summed across the participating regions, the
total predicted probability equals the total number of observed events. For this
step each client returns only three aggregate scalars — its sum of predicted
probabilities, its sum of `p(1-p)`, and its event count — so no patient-level
data and no regularisation parameter are shared
(`src/flower_fl.py:federated_intercept_recalibration`). The recalibration changes
only the intercept, leaving AUROC and the calibration slope unchanged, and is a
no-op for the maximum-likelihood local and centralized models. The reported
wall-clock time includes this step.

### Evaluation metrics
* **Discrimination** — AUROC.
* **Calibration slope** — the coefficient of a logistic regression of the
  observed outcome on the linear predictor (logit of the predicted probability);
  perfect = 1.
* **Calibration-in-the-large (intercept)** — the intercept of the same logistic
  model fitted with the linear predictor as an offset; perfect = 0.

Point estimates are reported with 95% bootstrap confidence intervals (1000
resamples) (`src/calibration.py`).

### Within-region validation (Figure 1, Table 2 / S3)
Stratified 5-fold CV per region. Local models train on 4/5 and test on 1/5.
The global models use a **leave-one-region-out (LORO)** scheme — for each target
region they are trained on the 15 other regions and evaluated on that region's
held-out 1/5 test folds, so no test data enter training
(`src/analysis.py:within_region`).

### Between-region validation (Figures 2–4 / S2–S10)
Every local model is applied to every other region, alongside the LORO global
models (Centralized and FL, where FL is the recalibrated federated model). The
local model with the largest performance deviation (Region 10) is highlighted.
The calibration-intercept panels (Figure 4 / S8–S10) additionally show the
uncorrected FedAvg model so the calibration-in-the-large offset removed by the
recalibration is visible.

### Full-parameter PCA (Figure 5)
Principal component analysis of the full parameters of the 18 models shown in the
manuscript: 16 local models, Centralized, and FL, where **FL** denotes the
recalibrated federated model. Each model contributes the 17 standardised slopes
**and the intercept** (calibration-in-the-large), expressed on a common
standardisation. Including the intercept makes the FL calibration-in-the-large
offset visible as a separation from the centralized model, with the recalibrated
FL sitting closer to it. The uncorrected FedAvg model is computed internally but
dropped from the final PCA output (`remap_pca`), so it does not appear in
Figure 5 (`src/analysis.py:coefficient_pca`, `src/output_utils.py:remap_pca`).

### FL convergence
Global training log-loss and AUROC are logged each round; the converged round,
the recalibration step, and the total wall-clock time are reported
(`FL_convergence` sheets and figure).

---

## Package layout

```
run_all.py                 orchestrates the whole analysis + exports
regen_figures.py           rebuild figures/Excel from saved results
src/
  config.py                predictors, regions, penalties, hyper-parameters
  data_utils.py            loading, region split, federated standardisation
  models.py                centralized/local estimators + CV
  flower_fl.py             Flower FedAvg + intercept recalibration + convergence
  analysis.py              within/between-region, full-parameter PCA, convergence
  calibration.py           AUROC + calibration (slope, calibration-in-the-large) + bootstrap
  excel_export.py          Excel in the figure-data layout
  figures.py               publication-format figures
flower_distributed/        networked Flower deployment (server + 16 clients)
```

The manuscript analyses are reproduced solely by the in-process implementation in
`src/flower_fl.py`. The `flower_distributed/` directory provides a networked
Flower demonstration of coefficient aggregation using the shared federated scaler,
but it is **not** the source of the manuscript results and does not implement the
full analysis pipeline: it omits the per-client cross-validated hyper-parameter
selection (clients take `--C` / `--l1_ratio` as arguments) and the federated
intercept recalibration, and it uses a different number of local iterations
(`--local_iters`, default 5, vs. one local update in the main analysis).
