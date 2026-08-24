# Robustness as a Tunable Design Objective for Lightweight IoT Intrusion Detection

Code and results for the paper of the same name, submitted to *Applied Sciences*.

The paper argues that clean-data accuracy is a poor guide to how an intrusion
detector behaves once it is deployed, and that the noise level used during
training can be treated as an explicit dial that sets a measurable robustness
radius. This repository contains everything needed to reproduce that argument:
every table, every figure, and every number quoted in the text.

---

## 1. What you need before starting

**Python 3.11**, 64-bit. No GPU is required. Every result in the paper was
produced on a single Intel Core i7-14700HX under Windows.

**The two datasets**, which are not redistributed here because they carry their
own licences.

| Dataset | Where to get it | File to download |
|---|---|---|
| ToN-IoT | University of New South Wales | `Train_Test_Network.csv` |
| Edge-IIoTset | IEEE DataPort | `DNN-EdgeIIoT-dataset.csv` |

Place both CSV files in a folder called `data/` next to `src/`. Nothing else
needs to be downloaded.

## 2. Setting up

```bash
git clone <this repository>
cd <repository>

python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # macOS or Linux

pip install -r requirements.txt
```

Then create the two folders the scripts write into:

```bash
mkdir data outputs
```

Copy the two CSV files into `data/`. Every script is run from inside `src/`:

```bash
cd src
python train.py
```

## 3. How the configuration works

**`config.py` is the only file you ever need to edit.** Every script imports it.
Three switches control which experiment you are running.

| Switch | Values | Meaning |
|---|---|---|
| `ACTIVE_DATASET` | `"edge_iiot"`, `"ton_iot"` | which dataset |
| `MODE` | `"binary"`, `"multiclass"` | normal against attack, or attack type |
| `TRAIN_NOISE_STD` | `0.0`, `0.05`, `0.1` | the training-noise level, the dial the paper is about |

Output files are named after `RUN_TAG`, which is assembled from
`ACTIVE_DATASET`, `MODE` and the optional suffix `EXP_NOTE`. When you train a
noise-aware model you must set `EXP_NOTE` as well, otherwise it will overwrite
the standard model.

The repository ships with the **main configuration**, which is what almost every
script expects:

```python
ACTIVE_DATASET = "edge_iiot"
MODE           = "binary"
K_FEATURES     = 1000     # 1000 means "keep all 49 features"
LOSS_TYPE      = "ce"
TRAIN_NOISE_STD = 0.0
CLIPNORM       = None
EXP_NOTE       = ""
```

If a script stops with a message about a dimension mismatch, the cause is almost
always that `K_FEATURES` was left at a small value from the feature-selection
ablation. Set it back to 1000.

## 4. The three models everything else depends on

Most scripts load trained models rather than training their own. Build these
three first, in this order. Each is one run of `train.py` with a different
`config.py`.

| Step | Set in `config.py` | Command | Saves |
|---|---|---|---|
| 1 | main configuration as shipped | `python train.py` | `model_edge_iiot_binary.keras` |
| 2 | `TRAIN_NOISE_STD = 0.05`, `EXP_NOTE = "noise05"` | `python train.py` | `model_edge_iiot_binary_noise05.keras` |
| 3 | `TRAIN_NOISE_STD = 0.1`, `EXP_NOTE = "noise"` | `python train.py` | `model_edge_iiot_binary_noise.keras` |

Then set `config.py` back to the main configuration before running anything
else. Roughly 30 to 45 minutes per model on a CPU.

## 5. Which script produces which table or figure

Run them in the order of the stages below. Within a stage the scripts are
independent of one another.

### Stage A. Data preparation and sanity checks

| Script | Produces | In the paper |
|---|---|---|
| `inspect_edge.py` | `edge_columns_report.txt` | the feature-engineering decisions, Section 3.3 |
| `sampling_check.py` | `sampling_check.json` | the stratification evidence, Section 3.1 |
| `check_preprocessing_integrity.py` | `preprocessing_integrity_*.json` | the no-leakage evidence, Section 3.2 |
| `rf_tuning.py` | `rf_tuning.json` | the random-forest search, Section 3.8 |

These are fast, none takes more than a few minutes.

### Stage B. Headline detection performance

| Script | Produces | In the paper |
|---|---|---|
| `train.py` | metrics, confusion matrix, report | Table 3, Figure 2, Figures S11 to S14 |
| `complexity_profile.py` | `complexity_profile.json` | Table 4 |
| `multiseed_baselines.py` | `multiseed_baselines.json` | Table 5, and the first rows of Table 7 |
| `baseline_transformer.py` | `transformer_baseline.json` | the transformer row of Table 5, Section 4.3.1 |
| `baselines.py` | `baselines_*.json` | single-seed version of Table 5, useful as a quick check |

`multiseed_baselines.py` takes five to seven hours and resumes if interrupted.
`baseline_transformer.py` takes four to six hours.

### Stage C. Feature selection

| Script | Produces | In the paper |
|---|---|---|
| `plot_feature_ablation.py` | `fig_feature_ablation.png` | Table 8, Figure S4 |
| `robustness_vs_k.py` | `robustness_vs_k.json` | Table 9, Figure S5 |

`plot_feature_ablation.py` needs `metrics_edge_iiot_binary_k{10,20,30,40,49}.json`,
which come from running `train.py` five times with `K_FEATURES` set to each value
and `EXP_NOTE` set to `"k10"` and so on.

### Stage D. The robustness results

| Script | Produces | In the paper |
|---|---|---|
| `robustness.py` | per-model noise and PGD curves | the curves behind Figure 4 |
| `robustness_tradeoff.py` | `robustness_tradeoff.json` | Figure 3, Figure S6 |
| `multiseed_robustness.py` | `multiseed_robustness.json` | Table 10 |
| `ablation_robustness.py` | `ablation_robustness.json` | Table 6 |
| `adversarial_training.py` | `advtrain_compare.json` | Table 11, Figure 8 |
| `robustness_ton_iot.py` | `robustness_ton_iot.json` | Figure 9 |
| `robustness_multiclass.py` | `robustness_multiclass_*.json` | Figure 10, Figure S7 |

`multiseed_robustness.py` and `ablation_robustness.py` retrain models on every
seed and take three to four hours each, both with resume support.
`adversarial_training.py` takes two to three hours because of its inner attack
loop. Run `robustness_multiclass.py` once per dataset.

### Stage E. The tunable radius and the decision threshold

| Script | Produces | In the paper |
|---|---|---|
| `sigma_radius.py` | `sigma_radius_*.json` | Figure 13, Figures S9 and S10 |
| `multiseed_radius.py` | `multiseed_radius.json` | Table 12, and the subset-variance number in Section 4.6.1 |
| `threshold_sweep.py` | `threshold_sweep.json` | Figures 14 and 15 |

Run `sigma_radius.py` once with `ACTIVE_DATASET = "edge_iiot"` and once with
`"ton_iot"`. It trains one model per noise level if that model is not already on
disk, so allow one and a half to two hours the first time.

### Stage F. Mechanism and additional attacks

| Script | Produces | In the paper |
|---|---|---|
| `feature_saliency_analysis.py` | `fig_feature_saliency.png` | Figure 6 |
| `rf_mechanism.py` | `rf_mechanism.json` | Section 4.6.1, data for Figure 5 |
| `margin_radius.py` | `margin_radius.json` | Section 4.6.2, data for Figure 7 |
| `robustness_adaptive.py` | `robustness_adaptive.json` | Figure 12 |
| `attacks_extra.py` | `attacks_extra.json` | Table 14, Section 4.6.8 |
| `correlated_perturbation.py` | `correlated_perturbation.json` | Table 13, Section 4.6.7 |

`margin_radius.py` requires the sigma-grid models that `sigma_radius.py` saves,
so run Stage E first.

### Stage G. Statistics and final figures

| Script | Requires | Produces | In the paper |
|---|---|---|---|
| `paired_tests.py` | `multiseed_baselines.json`, `multiseed_robustness.json` | `paired_tests.json` | the first rows of Table 7 |
| `paired_tests_robust.py` | `ablation_robustness.json` | `paired_tests_robust.json` | the remaining rows of Table 7 |
| `make_figures.py` | `rf_mechanism.json`, `margin_radius.json` | `fig_rf_mechanism.pdf`, `fig_margin_radius.pdf` | Figures 5 and 7 |

All three run in seconds because they only read json files.

## 6. Shared modules

Four files are imported by the scripts rather than run directly.

| Module | What it holds |
|---|---|
| `config.py` | every path, hyperparameter and switch |
| `data_preprocessing.py` | cleaning, encoding, scaling, splitting, feature selection, and the continuous-feature mask |
| `model.py` | the proposed detector, including the optional training-time noise layer |
| `attacks.py` | the single PGD implementation, the noise helper, and the radius extraction used everywhere |

`attacks.py` matters for reproducibility. Every reported attack in the paper
goes through the same function, so no result depends on a subtly different
implementation.

## 7. Reproducibility notes

**Seeds.** The seeds are 42, 7, 123, 2024 and 999 throughout. The radius
experiment uses the first three of these.

**What the perturbations touch.** All features are min-max scaled to [0, 1] and
only the features marked continuous by `data_preprocessing.py` are ever
perturbed, because altering a protocol flag or an identifier would produce
traffic that could not exist. The mask is carried through every attack.

**Attack budgets.** PGD budgets run from 0.005 to 0.05 and Gaussian noise levels
from 0.01 to 0.30, both expressed as a fraction of the observed range of each
feature.

**Where the randomness lives.** `data_preprocessing.py` uses `config.RANDOM_STATE`
for the subsample and for the train, validation and test split. Scripts that
sweep seeds therefore see a different subsample per seed, which is deliberate
and is stated in the paper wherever it affects a reported spread.

**Resuming.** The long multi-seed scripts write a checkpoint after every seed.
Restarting skips what is finished. Delete the `*_ckpt.json` file to force a full
rerun.

## 8. What is in `results/`

The json and text outputs behind the tables that were added during revision are
included so that the numbers in the paper can be checked without rerunning
anything. Model weights are not included, because they are large and every
script that needs one retrains it if it is missing.

## 9. Repository layout

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── data/                  you create this and put the two CSV files in it
├── outputs/               you create this, everything is written here
├── results/               json and text outputs from the runs reported in the paper
├── docs/
│   └── edge_columns_report.txt
└── src/
    ├── config.py                        edit this first
    ├── data_preprocessing.py
    ├── model.py
    ├── attacks.py
    ├── baselines.py
    ├── train.py
    └── ...                              the experiment scripts of Stages A to G
```

## 10. Citation

The BibTeX entry will be added once the paper is accepted.

## 11. Licence

MIT, see `LICENSE`. The datasets are covered by their own licences and are not
redistributed here.
