# Clinical readmission pipeline

Assessment 2: Applied MLOps pipeline project

Name: Darrell Ryman

Student ID: S00445230

Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

Client question: At the point of discharge, will an adult inpatient be readmitted within 30 days?

A Python pipeline that predicts whether an adult patient will be readmitted to hospital within 30 days. It trains on a synthetic clinical dataset of 10,000 patients. Rows outside a value range are removed at ingest. As shipped it is age 18 to 100 (`data.min_value`, `data.max_value` and `data.value_column`, all configurable, so you can filter another numeric column or study one band). The data is generated, so nothing here says anything about real patients. See [Limits](#limits).

Every setting lives in `config.yaml`: file paths, column names, feature rules, model settings and output names. Change behaviour there and leave the code alone.

## What it does

Each stage is its own program. They pass results to each other through files.

| Stage | Command | What it does |
|---|---|---|
| Ingest | `python -m src.ingest` | Loads the raw CSV from a file, a url or Kaggle (`data.source`), keeps the listed columns, cleans it, drops rows whose sex isn't Female or Male, removes patients outside the age range, sets aside a share of patients as "new" patients, saves a cleaned copy |
| Validate | `python -m src.validate` | Checks the cleaned data against the rules before anything uses it |
| Train | `python -m src.train` | Builds features, splits train and test, saves the encoded features for inspection, fits logistic regression, checks it against the approval gate, then saves the model and a model card |
| Evaluate | `python -m src.evaluate` | Scores the model on the test rows, writes the metrics file, and prints the error counts and the scores by sex and age band |
| Compare | `python -m src.compare` | Trains a baseline, a simple age rule and two candidate models on the same rows, writes one comparison table, and writes an evaluation evidence record |
| Approve | `python -m src.approve --name ... --role ... --decision approved --reason ...` | A named person records a sign-off for the saved model. `predict` and `serve` refuse to use a model without one. See [Approval gate and model card](#approval-gate-and-model-card) |
| Predict | `python -m src.predict` | Scores the new patients with the saved pipeline, and checks the predictions against their kept-apart answers |
| Plots | `python -m src.plots` | Optional. Draws four evidence graphs into `reports/figures/`: ROC and precision-recall curves, cross-validated AUC per fold, the error-count grid, and recall and precision by group. See [Graphs](#graphs) |
| Answer evidence | `python -m src.answers_evidence` | Run it last. Prints every assessment answer with the evidence that supports it, in pipeline order, into one report. Reads the data, the saved model, the comparison table and MLflow. Trains and writes nothing |
| Serve | `python -m src.serve` | Optional. Serves the saved model as a web API (FastAPI). See [Prediction service](#prediction-service) |

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
    new/              the set-aside new patients and their answers
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
    approve.py        records a named person's sign-off, and checks it before a model is used
    predict.py        scores new data
    serve.py          the prediction service (FastAPI), optional
    answers_evidence.py   the assessment answers, each below its evidence
    check_setup.py    reports which packages and programs are installed
  tests/              pytest tests
  artifacts/          the saved model and its model card
  reports/            metrics, comparison table, evidence record, predictions, RTF output
```

## Reproduce the results from a fresh clone

Follow these steps in order. Every command is typed into a terminal, from the folder named in each step. The commands below use Git Bash on Windows. On macOS or Linux, only step 4 differs.

**Step 1. Install Python and Git.** Skip this if both are already installed. Check by running:

```
python --version
git --version
```

Python should report 3.12 (the version the automated checks use). If either command isn't found, install it from python.org or git-scm.com and open a new terminal.

**Step 2. Download the code.** Run these two lines. The first makes a folder called `clinical_readmission` and the second moves into it. All later steps run from inside this folder.

```
git clone https://github.com/darrellryman/clinical_readmission.git
cd clinical_readmission
```

**Step 3. Create a virtual environment.** This keeps the project's libraries separate from the rest of the computer.

```
python -m venv .venv
```

**Step 4. Switch the virtual environment on.** Run the line for the system in use. The terminal prompt then starts with `(.venv)`. Repeat this step each time a new terminal is opened.

```
source .venv/Scripts/activate      # Windows, Git Bash
.venv\Scripts\activate             # Windows, PowerShell
source .venv/bin/activate          # macOS or Linux
```

**Step 5. Install the libraries.** This needs an internet connection and takes a few minutes. Run both lines. The second adds the MLflow web page files without pyarrow (there is no Windows on ARM build of it).

```
pip install -r requirements.txt
pip install mlflow==3.18.0 --no-deps --force-reinstall
```

**Step 6. Check the setup.** It should report nothing missing. If it prints a `pip` command, run that command, then run this step again.

```
python src/check_setup.py
```

On Windows on ARM, a few packages need pinning or a two-step install. The Git setup guide covers those.

**Step 7. Run the tests.** All of them should pass.

```
python -m pytest tests -q
```

**Step 8. Run the pipeline.** This downloads the data from Kaggle (no account needed), cleans it, trains and compares the models, and draws the graphs. It needs an internet connection. Choose one of the two ways. Both give the same results.

Way A, one command. The raw CSV is already in the repository, so DVC can start straight away. `--force` reruns every stage, because on a fresh clone DVC would otherwise report that everything is up to date.

```
dvc repro --force
```

Way B, six commands, one after the other:

```
python -m src.ingest
python -m src.validate
python -m src.train
python -m src.evaluate
python -m src.compare
python -m src.plots
```

**Step 9. Approve the model.** The prediction commands refuse to run until a named person has signed off. Replace the four values with the real person's details. This writes `approvals/signoff.json`.

```
python -m src.approve --name "Jane Smith" --role "Clinical owner" --decision approved --reason "Reviewed the model card and test results"
```

**Step 10. Predict on the new patients.** This scores the patients held back at the start and writes the predictions file.

```
python -m src.predict
```

**Step 11. Write the answers report.** This collects the evidence from every earlier step into one report.

```
python -m src.answers_evidence
```

If Way A was used in step 8, run this instead. It forces the report to refresh now that the sign-off and predictions exist:

```
dvc repro --force answers_evidence
```

**Step 12. Check the results.** Reports are written to the `reports` folder. Compare the run against the reference figures below.

The random seed and every setting live in `config.yaml`, and models run on one thread, so the same data, settings and library versions give the same numbers. You can check your run against these figures from the reference run:

| Result | Expected |
|---|---|
| Rows after cleaning | 9,196 (116 removed for sex "Other", 204 for age outside 18 to 100) |
| Training rows | 7,356 |
| Cross-validated AUC, logistic regression | 0.606, baseline 0.500, gain 0.106 (the gate needs 0.05) |
| Test AUC | logistic regression 0.639, random forest 0.636, age rule 0.511, baseline 0.500 |
| Test error counts (saved model) | 159 found, 99 missed, 552 false alarms, 1,030 correct |

Small differences in the third decimal can appear if library versions differ. `requirements.txt` sets minimum versions for most libraries and pins only the MLflow ones, so a newer scikit-learn may move a score slightly. Every output file is listed in [Output files](#output-files).

**Step 13. Prediction Service** To serve predictions over a web API once the model is approved, see [Prediction service](#prediction-service).

**Step 14. Track Runs With MLflow** To browse the logged runs, metrics and models in the MLflow web page, see [Track runs with MLflow](#track-runs-with-mlflow).

## Loading the raw data

A copy of the raw dataset is committed to this repository at `data/raw/synthetic_clinical_dataset.csv`, so a fresh clone has it and there is no DVC remote to set up. The data is a synthetic clinical dataset that mimics real patient records, from Kaggle: [Clinical Dataset](https://www.kaggle.com/datasets/uom190346a/synthetic-clinical-tabular-dataset).

Set `data.source` in `config.yaml` to choose where ingest gets it. There are three options. Whichever is chosen, the file ends up at `data.raw_path`, so everything after ingest, including DVC and MLflow, sees one raw file in one place.

**kaggle (the default).** Ingest downloads the file from Kaggle itself, using the `kagglehub` package that `requirements.txt` installs. No Kaggle account or login is needed for this public dataset.

```
data:
  source: kaggle
  kaggle_dataset: uom190346a/synthetic-clinical-tabular-dataset
  kaggle_file: synthetic_clinical_dataset.csv
```

Then run `python -m src.ingest`. If a different dataset asks for credentials, set `KAGGLE_USERNAME` and `KAGGLE_KEY` (see the kagglehub documentation) and never commit them.

**file.** Ingest reads a CSV that has already been downloaded by hand. A manual download from the Kaggle page needs a free Kaggle account.

```
data:
  source: file
  raw_path: data/raw/synthetic_clinical_dataset.csv
```

Download the CSV from the Kaggle page, save it at that path (rename it if the downloaded file has a different name), then run `python -m src.ingest`. To read a file from somewhere else for one run only, use `python -m src.ingest --input some/file.csv`, which overrides the config.

**url.** Ingest downloads the CSV from a direct web address (http or https only) and saves it at `data.raw_path`. The address has to point straight at the file. A normal Kaggle page link won't work, because Kaggle asks for a login.

```
data:
  source: url
  raw_url: https://example.com/path/to/synthetic_clinical_dataset.csv
```

Then run `python -m src.ingest`. Each attempt waits `data.download_timeout` seconds and tries up to `data.download_retries` times, pausing a little longer after each failure. The download is written to a temporary name first, so a failed or half finished download never replaces a good copy. If every attempt fails, ingest stops with a message naming the address and the last error.

Every option stops the run on an empty or header-only file. Each run prints the source type, the location, the time and a short fingerprint of the file at the top of `reports/ingest_output.rtf`. The same bytes always give the same fingerprint, so two runs can be compared, and a Kaggle download matches the same file downloaded by hand.

The Kaggle page doesn't state a licence or a file layout, so check the page's terms before reusing the data. If the file's columns differ from `data.keep_columns` in `config.yaml`, ingest and validate stop with a message naming what's missing.

## Configure

The settings you're most likely to touch:

```
data:
  source: kaggle                         # kaggle (default), file, or url (download raw_url first)
  raw_path: data/raw/synthetic_clinical_dataset.csv   # the raw file (a url download is saved here)
  raw_url: null                          # http or https address, used when source is url
  kaggle_dataset: uom190346a/synthetic-clinical-tabular-dataset   # owner/name, used when source is kaggle
  kaggle_file: synthetic_clinical_dataset.csv                      # the file inside that dataset
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
  require_signoff: true                  # predict.py and serve.py need a named person's sign-off
  signoff_path: approvals/signoff.json   # where src.approve writes it. Commit this file
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

One function, `dfEngineerFeatures` in `src/features.py`, builds these columns. It is the first step of the saved pipeline (see `objBuildFeatureStep`), so the model file carries its own feature step and training, `evaluate`, `predict.py` and the web service can't drift apart. You give the saved model raw patient columns and it builds the rest. The sex categories come from the config, and a value that isn't on the list stops the run with an error.

## Approval gate and model card

The prediction commands refuse to run until a named person has signed off. Replace the four values with the real person's details. This writes `approvals/signoff.json`.

```
python -m src.approve --name "Jane Smith" --role "Clinical owner" --decision approved --reason "Reviewed the model card and test results"
```

Read `artifacts/model_card.json` and `reports/evaluation_evidence.md` before signing. The sign-off file records the fingerprint of the model that was approved. `predict.py` and `serve.py` stop with `ModelNotApprovedError` if the file is missing, the decision is "rejected", or the model file no longer matches the fingerprint. A retrained model that comes out different needs a new sign-off. Commit the file to Git, so the history shows who approved what and when. It is a plain file, so a real system would sign it or keep it somewhere with access control.

***Set `approval.require_signoff` to false to switch the check off.***

Before any of that, `train.py` has to pass a gate. The model's cross-validated AUC on the training rows must beat the do-nothing baseline by at least `approval.min_auc_gain` (0.05). A model that fails isn't saved, the stage stops with `ModelGateError`, and it can't be approved. The threshold sits in the config because it should come from the intended use and the cost of errors. Passing the gate isn't clinical approval. The named person still decides.

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
python -m src.serve
```

Then open `http://127.0.0.1:8000/` for a form where you enter one patient's predictors and it answers "Possible readmission after 30 days: Yes" or "No". The form is built from `features.feature_columns` and `features.categorical_columns`, so it always matches the model's inputs. It starts with typical healthy adult values from `service.form_defaults` in `config.yaml` (assumed mg/dL units, not clinically signed off). Change them there, or set `{}` to start with blank boxes. The banner picture at the top is `assets/header_online.jpg`, set by `service.header_image`. Set it to `null` for no banner. Open `http://127.0.0.1:8000/docs` for a test page where you can send patients to the API directly. No Docker is needed. The host, port and the most patients per request are in the `service` section of `config.yaml`.

| Request | What it does |
|---|---|
| `GET /health` | Says the service is up and the model is loaded |
| `GET /info` | Returns the model card |
| `POST /predict` | Scores the patients in `{"patients": [{...}]}` and returns a class and a probability for each |

The request format is built from the config, so it always matches the model's inputs. A missing field, an extra field, a wrong type or a sex value that isn't on the list is refused with a 422 error before it reaches the model. Accepted patients go to the saved pipeline, which builds the engineered columns itself, as it did in training. The service stops at start-up if there is no saved model.

It is a demonstration. The model is trained on synthetic data and must not be used for real patients. The tests in `tests/test_serve.py` need the service libraries and are skipped if they aren't installed.

## Output files

By default, what each stage prints goes to an RTF file instead of the screen, for example `reports/train_output.rtf`. The name comes from `outputs.rtf_path` in the config (`{stage}` becomes the stage name). Tests write to a separate path set under `testing.rtf_path`. Each run overwrites that stage's file.

## Pipeline Output Files

Every file the solution writes when it runs, where it goes and what it holds. Locations are relative to the project root, and most of them come from `config.yaml`. The raw data file (`data/raw/synthetic_clinical_dataset.csv`) is an input, not an output, so it isn't listed.

| Produced by | File | Location | What it holds |
|---|---|---|---|
| **Data** | | | |
| ingest | `clean_synthetic_clinical_dataset.csv` | `data/processed/` | The cleaned patients used for training and testing: the 10 predictors, diagnosis and the readmission_30d answer. Rows with a sex other than Female or Male, or an age outside 18 to 100, are removed, and the set-aside new patients are taken out. 9,196 rows in the shipped run. |
| ingest | `new_synthetic_clinical_dataset.csv` | `data/new/` | The set-aside new patients (5 percent, 484 in the shipped run), with the same columns but no answer column. Never used for training or testing. |
| ingest | `new_synthetic_clinical_dataset_answers.csv` | `data/new/` | The real readmission_30d answers for the new patients, in the same row order. Used only to check the predictions. |
| train | `features.csv` | `data/processed/` | The encoded model inputs (including pulse_pressure and sex_Male), the target and a split column saying train or test. For inspection only. No stage reads it back. |
| **Model** | | | |
| train | `model.joblib` | `artifacts/` | The fitted pipeline, saved with joblib: the feature step (dfEngineerFeatures) and the logistic regression. Scores raw patient data on its own. |
| train | `model_card.json` | `artifacts/` | The model card: parameters, input and model columns, training rows, readmission rate, random seed, fingerprints of the model, data and config, Git commit, gate result, status, intended use, not-for list and known limits. |
| **Reports** | | | |
| evaluate | `metrics.json` | `reports/` | Test-row scores for the saved model: accuracy, F1, precision, recall and ROC AUC. |
| compare | `model_comparison.csv` | `reports/` | One row per model (baseline, age rule, logistic regression, random forest): role, type, cross-validated AUC mean and spread, and the test scores. |
| compare | `evaluation_evidence.md` | `reports/` | One-page evidence record: intended use, not-for list, results table, recommended model, limits and recommendation. |
| predict | `predictions.csv` | `reports/` | The new patients' columns plus a prediction column (the model's 0 or 1 for each patient). |
| ingest, validate, train, evaluate, compare, plots, predict | `{stage}_output.rtf (7 files, for example train_output.rtf)` | `reports/` | Everything each stage prints, saved as a Rich Text file instead of shown on screen. Each run overwrites its own file. |
| answers_evidence | `answer_evidence_output.rtf` | `reports/` | Every assessment answer in bold italic, each below the evidence that supports it, in pipeline order. Each run overwrites it. |
| **Graphs** | | | |
| plots | `1_roc_pr.png` | `reports/figures/` | ROC and precision-recall curves for each model. |
| plots | `2_cv_spread.png` | `reports/figures/` | Cross-validated AUC for each fold, by model. |
| plots | `3_confusion.png` | `reports/figures/` | The error-count grid for the saved model: readmissions found, missed and falsely flagged. |
| plots | `4_groups.png` | `reports/figures/` | Recall and precision by group (sex and age band). Drawn only when groups are set in the config. |
| **Run tracking and versioning** | | | |
| train, evaluate, compare | `mlflow.db` | `Project root` | The MLflow database of runs: settings, scores, and tags for the Git commit and the data, config and model fingerprints. |
| train, evaluate, compare | `mlruns/` | `Project root` | MLflow's copy of each run's files (model, model card, metrics, comparison table, evidence record), kept per run. |
| dvc repro | `dvc.lock` | `Project root` | The recorded hashes of every stage's inputs and outputs for the last run, so DVC can tell what changed. |
| **Test runs** | | | |
| pytest | `{stage}_output.rtf (5 files)` | `reports/test/` | The RTF output written by the tests, kept apart so a test run never overwrites a real run's reports. |

## Test

```
python -m pytest tests
```

212 tests. They use their own small config, so editing `config.yaml` for your data won't break them. One test checks that `config.yaml` itself is complete.

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

`.github/workflows/ci.yml` runs the tests on GitHub on every push to `main` and every pull request. It runs the tests only.

## Reproduce with DVC

`dvc.yaml` describes seven stages: ingest, validate, train, evaluate, compare, plots and answers_evidence.

Every stage output is marked `cache: false` in `dvc.yaml`. DVC still records each file's hash in `dvc.lock`, but the files themselves (raw and processed data, the saved model and card, metrics, reports and graphs) are committed to Git. There is no DVC remote and no DVC cache to lose. A clone from GitHub has everything.

On a fresh clone, skip the commands below. Run `dvc repro --force` to rebuild every stage, or `dvc repro` to check that nothing has changed.

To set DVC up in a new folder that isn't a clone, DVC needs a Git repository, so run `git init` first. With DVC installed, from the project root:

```
git init
dvc init
git add .dvc .dvcignore
git commit -m "Initialise DVC"
dvc repro
dvc metrics show
git add -A
git commit -m "Track the pipeline outputs"
```

If the folder is already a Git repository, skip `git init`. Git asks for a name and email on the first commit. Set them once with `git config user.name "Your Name"` and `git config user.email "you@example.com"`.

`dvc repro` reruns only the stages whose inputs changed. After a config change, `dvc metrics diff` shows how the scores moved. `predict` isn't a DVC stage, because it needs a new-patients file that you supply.

Without DVC, run the seven commands in the table above, in order.

## Predict on new patients

Ingest sets aside a share of patients (`data.new_data_fraction`, 5 percent by default) before anything else sees the data. They are chosen at random with the config seed, keeping the same readmission rate. Training, testing and comparing never touch them. Ingest writes them to two files, rebuilt on every run:

- `data/new/new_synthetic_clinical_dataset.csv`: the patients without the answer column
- `data/new/new_synthetic_clinical_dataset_answers.csv`: their real answers, in the same row order

Run `python -m src.approve` first if the model has no sign-off yet (see [Approval gate and model card](#approval-gate-and-model-card)). Then:

```
python -m src.predict
```

Predictions are written to `reports/predictions.csv`. Because these patients were never used for training, the run also prints scores against the kept-apart answers, so you can see how the model does on patients it has never met. Change the share in `config.yaml`, or set it to 0 to set none aside.

To score your own file, pass `python -m src.predict --input your_file.csv`. It needs the same columns as the new-patients file. The comparison with the answers file is skipped in that case. Column names must match the training data exactly, and any sex value other than Female or Male stops the run.

