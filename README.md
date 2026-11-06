# Clinical readmission pipeline

Assessment 2: Applied MLOps pipeline project

Name: Darrell Ryman

Student ID: S00445230

Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

Client question: At the point of discharge, will an adult inpatient be readmitted within 30 days?

A Python pipeline that predicts whether an adult patient will be readmitted to hospital within 30 days. 

It trains on a synthetic clinical dataset of 10,000 patients. Rows outside a value range are removed at ingest. As shipped it is age 18 to 100 (`data.min_value`, `data.max_value` and `data.value_column`, all configurable, so you can filter another numeric column or study one band). The data is generated, so nothing here says anything about real patients. See [Limits](#limits).

Every setting lives in `config.yaml`: file paths, column names, feature rules, model settings and output names. Change behaviour there and leave the code alone.

## What it does

Each stage is its own program. They pass results to each other through files.

| Stage | Command | What it does |
|---|---|---|
| Ingest | `python -m src.ingest` | Loads the raw CSV from a file or a url (`data.source`), keeps the listed columns, cleans it, drops rows whose sex isn't Female or Male, removes patients outside the age range, sets aside a share of patients as "new" patients, saves a cleaned copy |
| Validate | `python -m src.validate` | Checks the cleaned data against the rules before anything uses it |
| Train | `python -m src.train` | Builds features, splits train and test, saves the encoded features for inspection, fits logistic regression, checks it against the approval gate, then saves the model and a model card |
| Evaluate | `python -m src.evaluate` | Scores the model on the test rows, writes the metrics file, and prints the error counts and the scores by sex and age band |
| Compare | `python -m src.compare` | Trains a baseline, a simple age rule and two candidate models on the same rows, writes one comparison table, and writes an evaluation evidence record |
| Predict | `python -m src.predict` | Scores the new patients with the saved model, and checks the predictions against their kept-apart answers |
| Plots | `python -m src.plots` | Optional. Draws four evidence graphs into `reports/figures/`: ROC and precision-recall curves, cross-validated AUC per fold, the error-count grid, and recall and precision by group. See [Graphs](#graphs) |
| Serve | `python -m src.app` | Optional. Serves the saved model as a web API (FastAPI). See [Prediction service](#prediction-service) |

Run each one from the project root, in this order. Every program takes `--help`. A command line option such as `--input` or `--output` overrides the matching config value for that one run.

## Layout

```
clinical_readmission/
  README.md
  config.yaml         all settings
  dvc.yaml            the pipeline as DVC stages
  requirements.txt    core libraries, plus MLflow, FastAPI and matplotlib (optional, delete what you don't want)
  pytest.ini          tells pytest where to find src/
  .gitignore
  .github/workflows/ci.yml   runs the tests on GitHub
  data/
    raw/              the original CSV (never changed)
    processed/        the cleaned CSV (training and test patients) and features.csv, the encoded features
    new/              the set-aside new patients and their answers (not committed)
  src/
    config.py         loads and checks config.yaml
    display.py        dataset summary printing and RTF output
    ingest.py         stage 1
    validate.py       stage 2
    features.py       engineered features, X and y, train/test split
    train.py          builds and fits models, saves the model
    evaluate.py       metrics
    compare.py        baseline, rule and candidates, and the evidence record
    tracking.py       records runs in MLflow (optional)
    predict.py        scores new data
    app.py            the prediction service (FastAPI), optional
    check_setup.py    reports which packages and programs are installed
  tests/              pytest tests
  artifacts/          the saved model and its model card (not committed)
  reports/            metrics, comparison table, evidence record, predictions, RTF output (not committed)
```

## Setup

```
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
# macOS or Linux:  source .venv/bin/activate
# Windows PowerShell:  .venv\Scripts\activate
# Install the libraries. The file lists the core ones plus optional MLflow, FastAPI
# and matplotlib. Delete the lines you don't want first.
pip install -r requirements.txt
# Adds the MLflow web page files, without pyarrow (no Windows on ARM build). Skip it if you
# won't run mlflow ui.
pip install mlflow==3.18.0 --no-deps --force-reinstall
python src/check_setup.py
```

The last line lists what is installed and prints one `pip` command for anything missing. On Windows on ARM, a few packages need pinning or a two-step install. The Git setup guide covers those.

## Loading the raw data

`data.source` in `config.yaml` says where the raw data comes from, and ingest loads it the same way every run.

- `file` reads `data.raw_path`.
- `url` downloads `data.raw_url` (http or https only) into `data.raw_path`, then reads that copy. Everything after ingest, including DVC and MLflow, still sees one raw file in one place.

A download waits `download_timeout` seconds per attempt and tries up to `download_retries` times, pausing a little longer after each failure. It is written to a temporary name first, so a failed or half finished download never replaces a good copy. If every attempt fails, ingest stops with a message that names the address and the last error. An empty or header-only file also stops the run.

Each run prints the source type, the location, the time and a short fingerprint of the file at the top of `reports/ingest_output.rtf`. The same bytes always give the same fingerprint, so two runs can be compared. `python -m src.ingest --input some/file.csv` still reads a local file and overrides the config.

## Configure

The settings you're most likely to touch:

```
data:
  source: file                           # file, or url to download raw_url first
  raw_path: data/raw/synthetic_clinical_dataset.csv   # the raw file (a url download is saved here)
  raw_url: null                          # http or https address, used when source is url
  download_timeout: 30                   # seconds to wait on each download attempt
  download_retries: 3                    # attempts before ingest stops with an error
  keep_columns: [age, sex, bmi, ...]     # columns kept at ingest
  allowed_values:
    sex: [Female, Male]                  # rows with any other value are removed
  min_value: 18                            # lowest value kept, ends included (null leaves the low end open)
  max_value: 100                           # highest value kept (null leaves the high end open)
  value_column: age                      # the numeric column the range applies to
  new_data_fraction: 0.05                # share of patients set aside as new patients (0 for none)

features:
  feature_columns: [age, sex, bmi, ...]  # what the model sees
  target_column: readmission_30d
  test_size: 0.2
  stratify: true                         # keep the readmission rate the same in train and test
  categorical_columns:
    sex: [Female, Male]                  # becomes a 0/1 column, sex_Male
  difference_columns:
    pulse_pressure: [systolic_bp, diastolic_bp]   # systolic minus diastolic
  excluded_columns: [patient_id, mortality]       # the model must never use these

model:
  class_weight: balanced                 # readmissions are rare, so weight them up

evaluation:
  group_columns: [sex]                   # report scores for each sex
  binned_columns:
    age: [0, 40, 60, 80, 120]            # and for each age band

comparison:
  benchmarks:
    rule_age_65: {type: rule, column: age, threshold: 65}   # a fixed rule a model should beat

approval:
  enabled: true                          # train.py saves the model only if it passes the gate
  min_auc_gain: 0.05                     # cross-validated AUC must beat the baseline by this much
  intended_use: ...                      # text copied into the model card and evidence record

tracking:
  enabled: true                          # record runs in MLflow
  uri: sqlite:///mlflow.db
  experiment_name: clinical_readmission
```

Each setting has a comment in the file. Paths are relative to the project root. To use another config, pass `--config other.yaml` to any program or set the `PIPELINE_CONFIG` environment variable.

Two checks run when the config loads. A column listed in `excluded_columns` can't also be a feature, and a column used by a feature rule must be in `feature_columns`. Either mistake stops the run with a message.

## Features

There are 10 inputs: age, sex, BMI, systolic and diastolic blood pressure, glucose, cholesterol, creatinine, diabetes and hypertension, plus one built column, pulse pressure. Sex becomes the 0/1 column `sex_Male`.

`train.py` also writes the encoded features to `data/processed/features.csv` (`outputs.features_path`), with the target and a `split` column that says train or test, so you can open it and see exactly what the model saw. It is for inspection only. No stage reads it back, because every stage builds features with the same function, and reading a saved copy could let training and prediction drift apart.

One function, `dfEngineerFeatures` in `src/features.py`, builds these columns. Training and `predict.py` both call it, so the two can't drift apart. The sex categories come from the config, and a value that isn't on the list stops the run with an error.

## Results

On 1,840 held-out test patients. About 14 percent of patients are readmitted.

| Model | CV AUC | Test AUC | Accuracy | Recall | Precision |
|---|---|---|---|---|---|
| Baseline (always predicts "no") | 0.500 | 0.500 | 0.860 | 0.000 | 0.000 |
| Logistic regression | 0.606 | 0.639 | 0.646 | 0.616 | 0.224 |
| Rule: flag age 65 and over | 0.505 | 0.511 | 0.647 | 0.322 | 0.149 |
| Random forest | 0.599 | 0.636 | 0.653 | 0.612 | 0.227 |

CV AUC is the average over 5 cross-validation folds of the training rows.

What this says: the models find some signal and it is weak. Of the patients the model flags, about 22 percent are readmitted, against a base rate of 14. The baseline has the best accuracy because it never flags anyone, which is why recall and AUC matter here. The age rule is close to a coin flip, so age alone tells you almost nothing here. The random forest is within one standard deviation of logistic regression (0.020), so the gap is noise and the simpler model is the saved one. Pulse pressure and sex add almost nothing. Hypertension, diabetes and creatinine carry most of the signal.

A further 484 patients (5 percent) were set aside before training and never used for it or for these scores. Predicting on them gives accuracy 0.657, recall 0.662 and precision 0.239, in the same range as the test scores.

The table is rebuilt by `python -m src.compare` and saved to `reports/model_comparison.csv`.

## Approval gate and model card

`train.py` checks the model before saving it. Its cross-validated AUC on the training rows must beat the do-nothing baseline by at least `approval.min_auc_gain` (0.05). A model that fails isn't saved, and the stage stops with `ModelGateError`. The threshold sits in the config because it should come from the intended use and the cost of errors. Passing the gate isn't clinical approval. A named person still has to sign off.

A model that passes is saved with `artifacts/model_card.json`: the model type and settings, the input columns, the data, config and model fingerprints, the git commit, the gate result, the intended use and the known limits. `compare.py` writes `reports/evaluation_evidence.md`, which pulls the results, the recommended model, the limits and the recommendation into one page.

## Graphs

```
python -m src.plots
```

Run it after `train`. It writes four PNG files to `reports/figures/` (folder, sharpness and colours are in the `plots` section of `config.yaml`):

| File | Question it answers |
|---|---|
| `1_roc_pr.png` | Does the model find readmissions better than chance? |
| `2_cv_spread.png` | Is the gap between models bigger than the noise between folds? |
| `3_confusion.png` | How many readmissions are found, missed and falsely flagged? |
| `4_groups.png` | Are recall and precision even across sex and age band? |

The graphs use the same split, folds, models and group bands as `train`, `evaluate` and `compare`, so each number on a graph matches a number in the reports. The error grid and the group graph use the saved model. The curves and fold spread use the models in the `comparison` section.

## Prediction service

```
python -m src.app
```

Then open `http://127.0.0.1:8000/` for a form where you enter one patient's predictors and it answers "Possible readmission after 30 days: Yes" or "No". The form is built from `features.feature_columns` and `features.categorical_columns`, so it always matches the model's inputs. It starts with typical healthy adult values from `service.form_defaults` in `config.yaml` (assumed mg/dL units, not clinically signed off). Change them there, or set `{}` to start with blank boxes. The banner picture at the top is `assets/header.jpg`, set by `service.header_image`. Set it to `null` for no banner. Open `http://127.0.0.1:8000/docs` for a test page where you can send patients to the API directly. No Docker is needed. The host, port and the most patients per request are in the `service` section of `config.yaml`.

| Request | What it does |
|---|---|
| `GET /health` | Says the service is up and the model is loaded |
| `GET /info` | Returns the model card |
| `POST /predict` | Scores the patients in `{"patients": [{...}]}` and returns a class and a probability for each |

The request format is built from the config, so it always matches the model's inputs. A missing field, an extra field, a wrong type or a sex value that isn't on the list is refused with a 422 error before it reaches the model. Accepted patients go through `dfEngineerFeatures`, the same function training uses. The service stops at start-up if there is no saved model.

It is a demonstration. The model is trained on synthetic data and must not be used for real patients. The tests in `tests/test_app.py` need the service libraries and are skipped if they aren't installed.

## Assessment prompts in the reports

The stages print the assessment steps, in bold italic, into their RTF files. Each one starts `Step 1 - ` to `Step 6 - ` for the six assessment sections and sits just below the code that gives its evidence. They are `subPrintStep("question", "")` calls in `src/`. Type your answer between the empty quotes, then rerun the stage. A step with no answer prints `(answer not written yet)`, so gaps show up in the report.

## Output files

By default, what each stage prints goes to an RTF file instead of the screen, for example `reports/train_output.rtf`. The name comes from `outputs.rtf_path` in the config (`{stage}` becomes the stage name). Tests write to a separate path set under `testing.rtf_path`. Each run overwrites that stage's file.

## Test

```
python -m pytest tests
```

165 tests. They use their own small config, so editing `config.yaml` for your data won't break them. One test checks that `config.yaml` itself is complete.

## Track runs with MLflow

Train, evaluate and compare each record a run in MLflow: the settings, the scores, the model file and tags for the data fingerprint, the Git commit and the config fingerprint. Runs pile up, so you can see how results change between edits to `config.yaml`.

To look at the runs in your browser, run this after the Setup steps:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

If the page says "landing page (index.html) not found", the web page files are missing. The `pip install mlflow==3.18.0 --no-deps --force-reinstall` step in Setup puts them in place.

Open http://127.0.0.1:5000. Close any `mlflow ui` you started earlier first (Ctrl+C in its terminal). On Windows an old server can keep port 5000, and the browser may reach it instead of the new one. To trace a model, match its `model_fingerprint` tag, then read the `git_commit` and `data_fingerprint` tags on the same run.

Tracking is optional. Set `tracking.enabled: false` in `config.yaml`, or leave `mlflow-skinny` uninstalled, and the stages run as normal and say the run wasn't recorded. `mlflow.db` and `mlruns/` are not committed.

## Automated checks

`.github/workflows/ci.yml` runs the tests on GitHub on every push to `main` and every pull request. It runs the tests only. It can't run the pipeline, because the raw dataset is tracked by DVC and isn't in Git.

## Reproduce with DVC

`dvc.yaml` describes six stages: ingest, validate, train, evaluate, compare and plots. DVC needs a Git repository to start, so run `git init` first. With DVC installed, from the project root:

```
git init
dvc init
git add .dvc .dvcignore
git commit -m "Initialise DVC"
dvc add data/raw/synthetic_clinical_dataset.csv
git add data/raw/synthetic_clinical_dataset.csv.dvc data/raw/.gitignore
git commit -m "Track raw data with DVC"
dvc repro
dvc metrics show
```

If the folder is already a Git repository, skip `git init`. Git asks for a name and email on the first commit. Set them once with `git config user.name "Your Name"` and `git config user.email "you@example.com"`.

`dvc repro` reruns only the stages whose inputs changed. After a config change, `dvc metrics diff` shows how the scores moved. `predict` isn't a DVC stage, because it needs a new-patients file that you supply.

Without DVC, run the six commands in the table above, in order.

## Predict on new patients

Ingest sets aside a share of patients (`data.new_data_fraction`, 5 percent by default) before anything else sees the data. They are chosen at random with the config seed, keeping the same readmission rate. Training, testing and comparing never touch them. Ingest writes them to two files, rebuilt on every run:

- `data/new/new_synthetic_clinical_dataset.csv`: the patients without the answer column
- `data/new/new_synthetic_clinical_dataset_answers.csv`: their real answers, in the same row order

Then:

```
python -m src.predict
```

Predictions are written to `reports/predictions.csv`. Because these patients were never used for training, the run also prints scores against the kept-apart answers, so you can see how the model does on patients it has never met. Change the share in `config.yaml`, or set it to 0 to set none aside.

To score your own file, pass `python -m src.predict --input your_file.csv`. It needs the same columns as the new-patients file. The comparison with the answers file is skipped in that case. Column names must match the training data exactly, and any sex value other than Female or Male stops the run.

## Limits

- The data is synthetic. Results say nothing about real hospitals or patients.
- The model is weak. It is a worked example of the pipeline, not a tool for clinical decisions.
- The classification threshold is the default 0.5. Where to set it is a clinical choice about missed readmissions versus false alarms.
- 161 rows in the raw file have diastolic pressure at or above systolic. They are still in the data. The age filter is the only range check.
- Someone with clinical knowledge needs to confirm that each input is known at the moment the prediction would be made.
- `evaluate.py` prints scores by sex and age band. The data has no ethnicity or Aboriginal and Torres Strait Islander status column, so errors can't be checked for those groups. Cultural safety, data authority and consultation need to be handled before any real use.
- Leakage and train-serve risks, and what the pipeline does about each, are in `Leakage_And_Train_Serve_Risks.docx`.

## Notes

- `src/__init__.py` makes `src` a package so the programs can import each other.
- `data/`, `artifacts/` and `reports/` hold a `.gitkeep` so the folders exist in Git. Their contents are ignored.
- Add a library to `requirements.txt` the moment you import it.
