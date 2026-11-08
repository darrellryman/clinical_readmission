"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    answers_evidence.py

    The assessment answers, each with the evidence that supports it, in one place.

    The answers used to sit inside the stage files. They live here now, so the stages
    only run the pipeline and this file only reports on it. Each answer is printed
    just below the code that gives its evidence, in pipeline order: ingest, validate,
    train, evaluate, compare, predict and plots.

    Nothing is trained, fitted or overwritten here. The evidence is read from the
    cleaned data, the saved model and its model card, the comparison table and
    MLflow, or rebuilt in memory with the same functions the stages use. Run the
    pipeline first, so those files exist.

    The printed output is saved to an RTF file instead of the screen. The file name
    comes from outputs.rtf_path in config.yaml (the text {stage} becomes the stage name).

    Run from the project root, after the other stages:
        python -m src.answers_evidence
        python -m src.answers_evidence --input some/other/file.csv     (overrides the config)
"""

# argparse gives the script a small command line interface (--input, --config, --help).
import argparse

# json reads the model card. platform and importlib.metadata report the Python and
# library versions. subprocess asks Git for the commit history.
import importlib.metadata as objMeta
import json
import platform
import subprocess

# Path handles file paths the same way on Windows, macOS and Linux.
from pathlib import Path

import numpy as np

# pandas builds the small evidence tables.
import pandas as pd

# yaml reads dvc.lock.
import yaml

from sklearn.metrics import roc_auc_score

from src.config import subAddConfigArgument, dctLoadConfig
from src.display import (objCaptureToRtf, subPrintStep, subPrintPiplineHeader, bDisplayDatasetDetails, subPrintImage,
                         STR_BOLD, STR_UNDERLINE, STR_RESET, STR_ITALIC)
from src.evaluate import dctComputeMetrics, dfBuildGroups, dfConfusionCounts, dfMetricsByGroup
from src.features import tplBuildFeatures, lstSplitData, dfRawInputs
from src.ingest import dfLoadCsv, dfKeepColumns, dfCleanData, dfFilterAllowedValues, dfFilterValueRange
from src.tracking import strFileFingerprint, strGitCommit
from src.train import mdlLoadModel
from src.validate import dfValidate


# ---------------------------------------------------------------------------
# Evidence functions. Each one prints the figures an answer quotes.
# ---------------------------------------------------------------------------

def subPrintEvaluationStrategyEvidence(dctConfig: dict, serYTrain: pd.Series, serYTest: pd.Series) -> None:
    """Print the evaluation strategy evidence.

    Call it in a stage, just below the line of code that gives the evidence for
    the step. Because it uses print(), the text goes to the stage's RTF file
    with everything else.

    Args:
        dctConfig: The loaded config.yaml settings.
        serYTrain: The training targets from lstSplitData.
        serYTest: The test targets from lstSplitData.

    Returns:
        None. It only prints.
    """
    fltTest = dctConfig["features"]["test_size"]
    print(f"\n{STR_BOLD}{STR_UNDERLINE}Evaluation strategy{STR_RESET}")
    print(
        f"{STR_BOLD}Set aside as new patients:{STR_RESET} {dctConfig['data']['new_data_fraction']:.0%} of the cleaned rows")
    print(f"{STR_BOLD}Train / test split of the rest:{STR_RESET} {1 - fltTest:.0%} / {fltTest:.0%}")
    print(f"{STR_BOLD}Stratified on the target:{STR_RESET} {dctConfig['features']['stratify']}")
    print(f"{STR_BOLD}Readmission rate:{STR_RESET} train {serYTrain.mean():.1%}, test {serYTest.mean():.1%}")
    print(
        f"{STR_BOLD}Cross-validation folds on the training rows:{STR_RESET} {dctConfig['comparison']['cv_folds']}")
    print(f"{STR_BOLD}Random seed:{STR_RESET} {dctConfig['project']['random_state']}")


def subPrintGroupErrorEvidence(dfByGroup: pd.DataFrame, dctConfig: dict) -> None:
    """Print the scores by group that the group error answer quotes.

    Call it in a stage, just below the line of code that gives the evidence for
    the step. Because it uses print(), the text goes to the stage's RTF file
    with everything else.

    Args:
        dfByGroup: The table from dfMetricsByGroup, one row per group.
        dctConfig: The loaded config.yaml settings.

    Returns:
        None. It only prints.
    """
    dctEvaluation = dctConfig["evaluation"]
    lstChecked = list(dctEvaluation["group_columns"] or []) + list((dctEvaluation["binned_columns"] or {}).keys())

    print(f"\n{STR_BOLD}{STR_UNDERLINE}Group error evidence{STR_RESET}")
    print(f"\n{STR_BOLD}Groupings checked:{STR_RESET} {', '.join(lstChecked)}")
    print(dfByGroup[["group", "rows", "flagged_rate", "recall", "roc_auc"]].round(3).to_string(index=False))

    serLowest = dfByGroup.loc[dfByGroup["recall"].idxmin()]
    print(f"\n{STR_BOLD}Lowest recall:{STR_RESET} {serLowest['group']}, {serLowest['recall']:.3f}, "
          f"{int(serLowest['rows'])} test rows, about {round(serLowest['rows'] * serLowest['actual_rate'])} readmissions")
    print(f"\n{STR_BOLD}Flagged rate across all groups:{STR_RESET} "
          f"{dfByGroup['flagged_rate'].min():.0%} to {dfByGroup['flagged_rate'].max():.0%}")

def subPrintConsequencesEvidence(dfCounts: pd.DataFrame) -> None:
    """Print the error counts behind the false positive and false negative answer.

    Call it in a stage, just below the line of code that gives the evidence for
    the step. Because it uses print(), the text goes to the stage's RTF file
    with everything else.

    Args:
        dfCounts: The one-row table from dfConfusionCounts, with true_positive,
            false_negative, false_positive and true_negative.

    Returns:
        None. It only prints.
    """
    dctRow = dfCounts.iloc[0].to_dict()
    intTp, intFn = int(dctRow["true_positive"]), int(dctRow["false_negative"])
    intFp, intTn = int(dctRow["false_positive"]), int(dctRow["true_negative"])
    intReadmitted = intTp + intFn
    intFlagged = intTp + intFp

    print(f"\n{STR_BOLD}{STR_UNDERLINE}Error consequences evidence{STR_RESET}")
    print(f"\n{STR_BOLD}Readmissions in the test rows:{STR_RESET} {intReadmitted}")
    print(f"\n{STR_BOLD}False negatives (readmissions missed):{STR_RESET} {intFn} of {intReadmitted}")
    print(f"\n{STR_BOLD}Patients flagged:{STR_RESET} {intFlagged}")
    print(f"\n{STR_BOLD}False positives (flagged, not readmitted):{STR_RESET} {intFp}")
    print(f"\n{STR_BOLD}Share of flagged patients who were false alarms:{STR_RESET} {intFp / intFlagged:.1%}")


def subPrintMetricChoiceEvidence(dctConfig: dict, serY: pd.Series, serYTest: pd.Series, dctMetrics: dict) -> None:
    """Print the figures behind the choice of metrics.

    Call it in a stage, just below the line of code that gives the evidence for
    the step. Because it uses print(), the text goes to the stage's RTF file
    with everything else.

    Args:
        dctConfig: The loaded config.yaml settings. Uses evaluation.positive_label.
        serY: The target for all the cleaned rows.
        serYTest: The target for the test rows.
        dctMetrics: The scores from dctComputeMetrics for the saved model.

    Returns:
        None. It only prints.
    """
    varPositive = dctConfig["evaluation"]["positive_label"]

    # A model that predicts "no readmission" for everyone is right whenever the patient was not readmitted.
    fltAlwaysNo = float((serYTest != varPositive).mean())
    # A do-nothing model gives every patient the same score, so it can't rank anyone: AUC 0.5.
    fltBaselineAuc = float(roc_auc_score(serYTest, np.zeros(len(serYTest))))

    print(f"\n{STR_BOLD}{STR_UNDERLINE}Metric choice evidence{STR_RESET}")
    print(f"\n{STR_BOLD}Patients readmitted (all cleaned rows):{STR_RESET} {(serY == varPositive).mean():.1%}")
    print(f"\n{STR_BOLD}Accuracy if every patient is predicted 'no readmission' (test rows):{STR_RESET} {fltAlwaysNo:.1%}")
    print(f"\n{STR_BOLD}Do-nothing baseline AUC (test rows):{STR_RESET} {fltBaselineAuc:.3f}")
    print(f"\n{STR_BOLD}{STR_UNDERLINE}Logistic regression, test rows{STR_RESET}")
    print(f"\n{STR_BOLD}AUC:{STR_RESET} {dctMetrics['roc_auc']:.3f}")
    print(f"\n{STR_BOLD}Recall:{STR_RESET} {dctMetrics['recall']:.3f}")
    print(f"\n{STR_BOLD}Precision:{STR_RESET} {dctMetrics['precision']:.3f}")
    print(f"\n{STR_BOLD}Accuracy:{STR_RESET} {dctMetrics['accuracy']:.3f}")


def subPrintRunHistoryEvidence(dctConfig: dict, intLatest: int = 10) -> None:
    """Print the MLflow run history, then the latest run of each stage in detail.

    Call it in a stage, just below the line of code that gives the evidence for
    the step. Because it uses print(), the text goes to the stage's RTF file
    with everything else.

    Args:
        dctConfig: The loaded config.yaml settings. Uses tracking.enabled,
            tracking.uri and tracking.experiment_name.
        intLatest: How many of the most recent runs to list.

    Returns:
        None. It only prints. If tracking is off, MLflow is missing or no runs
        exist yet, it says so and carries on.
    """
    dctTracking = dctConfig["tracking"]
    print(f"\n{STR_BOLD}{STR_UNDERLINE}MLflow run history{STR_RESET}")
    if not dctTracking["enabled"]:
        print(f"\nMLflow tracking is switched off in config.yaml.")
        return

    try:
        import mlflow
        mlflow.set_tracking_uri(dctTracking["uri"])
        dfRuns = mlflow.search_runs(experiment_names=[dctTracking["experiment_name"]],
                                    order_by=["start_time DESC"], max_results=200)
        objClient = mlflow.MlflowClient()
    except Exception as objError:  # evidence must not stop the stage
        print(f"\nCould not read the MLflow history ({type(objError).__name__}: {objError}).")
        return

    if dfRuns.empty:
        print(f"\nNo runs recorded yet.")
        return

    # The list of recent runs: when, which stage, which code and config, and the AUC where there is one.
    dfShow = pd.DataFrame({
        "started (UTC)": pd.to_datetime(dfRuns["start_time"]).dt.strftime("%Y-%m-%d %H:%M"),
        "stage": dfRuns["tags.stage"],
        "run": dfRuns["tags.mlflow.runName"],
        "git commit": dfRuns["tags.git_commit"].str[:7],
        "config": dfRuns["tags.config_fingerprint"],
        "roc_auc": dfRuns["metrics.roc_auc"].round(3) if "metrics.roc_auc" in dfRuns else None,
    })
    print(f"\n{STR_BOLD}Experiment:{STR_RESET} {dctTracking['experiment_name']}, "
          f"{min(intLatest, len(dfRuns))} most recent of {len(dfRuns)} runs shown\n")
    print(dfShow.head(intLatest).to_string(index=False))

    # The newest run of each stage, in full: its settings, its fingerprints and the files attached.
    for strStage in ["train", "evaluate", "compare"]:
        dfStage = dfRuns[dfRuns["tags.stage"] == strStage]
        if dfStage.empty:
            continue
        objRun = objClient.get_run(dfStage.iloc[0]["run_id"])
        print(f"\n{STR_BOLD}{STR_UNDERLINE}Latest {strStage} run: {objRun.info.run_name}{STR_RESET}")
        print(f"\n{STR_BOLD}Settings:{STR_RESET} " + "; ".join(f"{strKey}={varValue}" for strKey, varValue in sorted(objRun.data.params.items())))
        for strTag in ["git_commit", "raw_data_fingerprint", "data_fingerprint", "config_fingerprint", "model_fingerprint"]:
            if strTag in objRun.data.tags:
                print(f"\n{STR_BOLD}{strTag}:{STR_RESET} {objRun.data.tags[strTag]}")
        lstFiles = [objFile.path for objFile in objClient.list_artifacts(objRun.info.run_id)]
        print(f"\n{STR_BOLD}Files attached:{STR_RESET} {', '.join(lstFiles)}")


def subPrintVersionEvidence() -> None:
    """Print the Git, config fingerprint, pythin version and DVC details.

    Call it in a stage, just below the line of code that gives the evidence for
    the step. Because it uses print(), the text goes to the stage's RTF file
    with everything else.

    Args:
        None.

    Returns:
        None. It only prints.
    """

    print(f"\n{STR_BOLD}{STR_UNDERLINE}Tracking evidence{STR_RESET}")
    print(f"{STR_BOLD}Git commit:{STR_RESET} {strGitCommit()}")
    print(f"{STR_BOLD}config.yaml fingerprint:{STR_RESET} {strFileFingerprint('config.yaml')}")
    print(f"{STR_BOLD}Python:{STR_RESET} {platform.python_version()}")

    print(f"\n{STR_BOLD}Library versions installed:{STR_RESET}")
    for strPackage in ["pandas", "scikit-learn", "joblib", "pyyaml", "mlflow-skinny", "dvc"]:
        try:
            strVersion = objMeta.version(strPackage)
        except objMeta.PackageNotFoundError:
            strVersion = "not installed"
        print(f"  {strPackage}: {strVersion}")

    print(f"\n{STR_BOLD}DVC lock file (dvc.lock):{STR_RESET}")
    try:
        with open("dvc.lock", encoding="utf-8") as objHandle:
            dctLock = yaml.safe_load(objHandle)
        for strStage, dctStage in dctLock["stages"].items():
            print(
                f"  {strStage}: {len(dctStage.get('deps', []))} inputs, {len(dctStage.get('outs', []))} outputs recorded")
        for dctOut in dctLock["stages"]["train"]["outs"]:
            print(f"  train output {dctOut['path']}: md5 {dctOut['md5'][:8]}")
    except (OSError, KeyError, TypeError):
        print("  No dvc.lock yet. Run dvc repro.")


def subPrintModelProvenanceEvidence(pthCard: Path, pthOut: Path, strInputPath: str) -> None:
    """Print where the model came from, read back from its model card.

    Call it in a stage, just below the line of code that gives the evidence for
    the step. Because it uses print(), the text goes to the stage's RTF file
    with everything else.

    Args:
        pthCard: The saved model card.
        pthOut: The saved model file.
        strInputPath: The processed CSV the model was trained from.

    Returns:
        None. It only prints.
    """

    dctSaved = json.loads(Path(pthCard).read_text(encoding="utf-8"))

    print(f"\n{STR_BOLD}{STR_UNDERLINE}Model provenance (from the model card){STR_RESET}")
    print(f"{STR_BOLD}Created (UTC):{STR_RESET} {dctSaved['created_utc']}")
    print(f"{STR_BOLD}Git commit:{STR_RESET} {dctSaved['git_commit']}")
    print(f"{STR_BOLD}Random seed:{STR_RESET} {dctSaved['random_state']}")
    print(f"{STR_BOLD}Gate result:{STR_RESET} {'PASSED' if dctSaved['gate']['passed'] else 'FAILED'}")
    print(f"{STR_BOLD}Status:{STR_RESET} {dctSaved['status']}")

    print(f"\n{STR_BOLD}Fingerprint check (card against the files on disk){STR_RESET}")
    for strLabel, strKey, varFile in [
        ("Model", "model_fingerprint", pthOut),
        ("Data", "data_fingerprint", strInputPath),
        ("Config", "config_fingerprint", "config.yaml"),
    ]:
        strNow = strFileFingerprint(varFile)
        strResult = "MATCH" if strNow == dctSaved[strKey] else "DIFFERENT"
        print(f"  {strLabel}: card {dctSaved[strKey]}, file {strNow}, {strResult}")


def subPrintDataSplitEvidence(dctConfig: dict, dfXTrain: pd.DataFrame, dfXTest: pd.DataFrame) -> None:
    """Print the data split evidence.

    Call it in a stage, just below the line of code that gives the evidence for
    the step. Because it uses print(), the text goes to the stage's RTF file
    with everything else.

    Args:
        dctConfig: The loaded config.yaml settings.

    Returns:
        None. It only prints.

    """
    # Split percentages: the settings from config.yaml, then the actual rows.
    fltNew = dctConfig['data']['new_data_fraction']
    fltTest = dctConfig['features']['test_size']
    intTrain, intTest = len(dfXTrain), len(dfXTest)
    intUsed = intTrain + intTest

    print(f"\n{STR_BOLD}{STR_UNDERLINE}Data split{STR_RESET}")
    print(
        f"{STR_BOLD}Set aside as new patients (data.new_data_fraction):{STR_RESET} {fltNew:.0%} of the cleaned rows")
    print(f"{STR_BOLD}Test share (features.test_size):{STR_RESET} {fltTest:.0%} of the remaining rows")
    print(f"{STR_BOLD}Stratified on the target (features.stratify):{STR_RESET} {dctConfig['features']['stratify']}")
    print(f"{STR_BOLD}Training rows:{STR_RESET} {intTrain} ({intTrain / intUsed:.1%} of the {intUsed} rows used)")
    print(f"{STR_BOLD}Test rows:{STR_RESET} {intTest} ({intTest / intUsed:.1%} of the {intUsed} rows used)")
    print(f"{STR_BOLD}Overall, as a share of the cleaned data:{STR_RESET} "
          f"train {(1 - fltNew) * (1 - fltTest):.1%}, test {(1 - fltNew) * fltTest:.1%}, new patients {fltNew:.1%}")


def tplIngestCounts(dctConfig: dict, dfRaw: pd.DataFrame) -> tuple:
    """Repeat the ingest cleaning steps in memory and count the rows each one removes.

    Nothing is written. It uses the same functions as ingest.py, in the same order,
    so the counts match the ingest report.

    Args:
        dctConfig: The loaded config.yaml settings. Uses data.keep_columns,
            data.allowed_values, data.value_column, data.min_value and data.max_value.
        dfRaw: The raw data.

    Returns:
        A tuple of four counts: raw rows, rows after duplicates are removed, rows
        after the allowed-value filter and rows after the range filter.
    """
    dctData = dctConfig["data"]
    dfWork = dfCleanData(dfKeepColumns(dfRaw, dctData["keep_columns"]))
    intAfterClean = len(dfWork)
    dfWork = dfFilterAllowedValues(dfWork, dctData["allowed_values"] or {})
    intAfterValues = len(dfWork)
    dfWork = dfFilterValueRange(dfWork, dctData["value_column"], dctData["min_value"], dctData["max_value"])
    return len(dfRaw), intAfterClean, intAfterValues, len(dfWork)


def subPrintRawDataEvidence(dctConfig: dict) -> None:
    """Print where the raw data came from.

    Call it in a stage, just below the line of code that gives the evidence for
    the step. Because it uses print(), the text goes to the stage's RTF file
    with everything else.

    Args:
        dctConfig: The loaded config.yaml settings. Uses data.source, data.raw_url
            and data.raw_path.

    Returns:
        None. It only prints.
    """
    dctData = dctConfig["data"]
    strLocation = dctData["raw_url"] if dctData["source"] == "url" else dctData["raw_path"]
    print(f"\n{STR_BOLD}{STR_UNDERLINE}Data source{STR_RESET}")
    print(f"{STR_BOLD}Source type:{STR_RESET} {dctData['source']}")
    print(f"{STR_BOLD}Location:{STR_RESET} {strLocation}")
    print(f"{STR_BOLD}Local copy read:{STR_RESET} {dctData['raw_path']}")
    print(f"{STR_BOLD}File fingerprint (SHA-256, first 12):{STR_RESET} {strFileFingerprint(dctData['raw_path'])}")


def subPrintTrainingEvidence(dctConfig: dict) -> None:
    """Print the training rows and approval gate result, read back from the model card.

    Call it in a stage, just below the line of code that gives the evidence for
    the step. Because it uses print(), the text goes to the stage's RTF file
    with everything else.

    Args:
        dctConfig: The loaded config.yaml settings. Uses approval.model_card_path.

    Returns:
        None. It only prints. If there is no model card it says so.
    """
    pthCard = Path(dctConfig["approval"]["model_card_path"])
    print(f"\n{STR_BOLD}{STR_UNDERLINE}Training evidence (from the model card){STR_RESET}")
    if not pthCard.is_file():
        print(f"No model card at {pthCard}. Run python -m src.train first.")
        return
    dctSaved = json.loads(pthCard.read_text(encoding="utf-8"))
    dctGate = dctSaved["gate"]
    print(f"{STR_BOLD}Training rows:{STR_RESET} {dctSaved['training_rows']}")
    if dctGate["enabled"]:
        print(f"{STR_BOLD}Cross-validated AUC:{STR_RESET} baseline {dctGate['baseline_cv_auc']:.3f}, "
              f"model {dctGate['model_cv_auc']:.3f}, gain {dctGate['auc_gain']:.3f}, "
              f"required {dctGate['required_gain']:.3f}, {'PASSED' if dctGate['passed'] else 'FAILED'}")
    else:
        print("The approval gate was switched off for this model.")


def subPrintComparisonEvidence(dctConfig: dict) -> None:
    """Print the model comparison table written by compare.py.

    Call it in a stage, just below the line of code that gives the evidence for
    the step. Because it uses print(), the text goes to the stage's RTF file
    with everything else.

    Args:
        dctConfig: The loaded config.yaml settings. Uses outputs.comparison_path
            and comparison.cv_folds.

    Returns:
        None. It only prints. If the table does not exist it says so.
    """
    pthTable = Path(dctConfig["outputs"]["comparison_path"])
    print(f"\n{STR_BOLD}{STR_UNDERLINE}Model comparison evidence{STR_RESET}")
    if not pthTable.is_file():
        print(f"No comparison table at {pthTable}. Run python -m src.compare first.")
        return
    print(f"\n{STR_BOLD}Cross-validation folds:{STR_RESET} {dctConfig['comparison']['cv_folds']}\n")
    print(pd.read_csv(pthTable).round(3).to_string(index=False))


def subPrintGraphEvidence(dctConfig: dict) -> None:
    """Print the graph files that plots.py wrote.

    Call it in a stage, just below the line of code that gives the evidence for
    the step. Because it uses print(), the text goes to the stage's RTF file
    with everything else.

    Args:
        dctConfig: The loaded config.yaml settings. Uses plots.folder.

    Returns:
        None. It only prints. If there are no graphs it says so.
    """
    pthFolder = Path(dctConfig["plots"]["folder"])
    lstFiles = sorted(pthFolder.glob("*.png")) if pthFolder.is_dir() else []
    print(f"\n{STR_BOLD}{STR_UNDERLINE}Graph evidence{STR_RESET}")
    if not lstFiles:
        print(f"No graphs in {pthFolder}. Run python -m src.plots first.")
    for pthFile in lstFiles:
        print(f"  {pthFile}")

    for pthFile in lstFiles:
        print(f"\n{STR_BOLD}{pthFile.name}{STR_RESET}")
        subPrintImage(pthFile)


# ---------------------------------------------------------------------------
# One section for each stage, in pipeline order.
# ---------------------------------------------------------------------------

def subAnswerIngest(dctConfig: dict) -> None:
    """Print the ingest answers, each below its evidence.

    Args:
        dctConfig: The loaded config.yaml settings.

    Returns:
        None. It only prints.
    """
    print(f"\n{STR_BOLD}{STR_UNDERLINE}ANSWERS: INGEST{STR_RESET}")

    # The raw file and the row counts at each cleaning step, rebuilt in memory.
    dfRaw = dfLoadCsv(dctConfig["data"]["raw_path"])
    intRawRows, intAfterClean, intAfterValues, intCleanRows = tplIngestCounts(dctConfig, dfRaw)
    subPrintRawDataEvidence(dctConfig)

    print(f"\n{STR_BOLD}{STR_UNDERLINE}Raw Data{STR_RESET}")
    bDisplayDatasetDetails(dfRaw)

    # The profile above is the evidence for the raw data and where it came from.
    subPrintStep(
        f"\n{STR_BOLD}Step 1 - {STR_RESET}{STR_ITALIC}Pipeline architecture diagram: describe the raw data and its source.{STR_RESET}",
        "The raw data is stored in one CSV file called synthetic_clinical_dataset.csv, which is defined in the config in data.raw_path. The data includes 10,000 rows (one per patient) and 14 columns in total. Six columns are integers, six are decimals, and two are text (sex and diagnosis). The data profile shows no missing values and no duplicate rows. The input variables for the tool are age, sex, BMI, systolic and diastolic blood pressure, glucose, cholesterol, creatinine, diabetes and hypertension. Diagnosis is a text column that stays in the cleaned file but is not used as an input. The target variable is readmission_30d, indicating an average of about 14% of patients were readmitted within 30 days. Mortality is a competing endpoint and would leak information if used as an input variable. The patient_id variable is a label and thus not used as an input variable. The data profile also reveals a problem that will surface later. Age has a minimum value of 0, meaning some records must be for newborns or children. The tool is designed to work with adult inpatients and thus the ingest filters on age.",
    )

    # dfLoadCsv above loads the raw data from the path in config.yaml, the same way every run.
    subPrintStep(
        f"\n{STR_BOLD}Step 2 - {STR_RESET}{STR_ITALIC}Implement a repeatable process for loading the raw data from a file, URL or other defined source.{STR_RESET}",
        "One function, tplLoadRawData, loads the raw data and reads its settings from the config, so every run loads the data the same way. data.source determines the data source (i.e., whether it is a file or a web address). For files, the raw data is read from data.raw_path. For web addresses, the raw data is downloaded from data.raw_url to the specified path. The download is saved with a temporary file name. Only when the download has completed is the temporary file name replaced with the final file name. Should the download fail for some reason or the resulting file be empty, the run is terminated with a message indicating the problem with the web address or file specified for the test case. In the run report, each run is described by the data source, the location from which the data was loaded, the time it was loaded, and a short file fingerprint. In this case, a local file was read as specified in the config.",
    )

    # dfLoadCsv above reads one file at a fixed path. Say what happens if that file is late, missing or empty.
    subPrintStep(
        f"\n{STR_BOLD}Step 6 - {STR_RESET}{STR_ITALIC}Identify operational risks, such as unavailable or delayed data.{STR_RESET}",
        "The biggest risk is that the raw data for a given file is missing or late. For files the system tries to download, if the file or associated web address is unavailable, or if the file is empty, the system will retry the download a number of times, then stop and provide a clear message. Late data is a gap that currently is not addressed. While load time and a file fingerprint are recorded after ingestion, nothing currently checks how old the file is; as a result, a file could be stale and used without warning. A changed layout is a third risk. If the dataset column changes, the Ingest stops when it can’t find a column listed in the config.",
    )

    # The raw data above is where data authority starts. Say who owns this data and whose consultation would be needed.
    subPrintStep(
        f"\n{STR_BOLD}Step 6 - {STR_RESET}{STR_ITALIC}Consider cultural safety, data authority and the need for consultation where the data, prediction or proposed use could affect Aboriginal and Torres Strait Islander peoples, communities, Country or knowledge.{STR_RESET}",
        "The dataset lacks a column (or combination of columns) that allows identification of Indigenous people. Therefore, it is not possible to evaluate how the model is treating Aboriginal and Torres Strait Islander patients. The dataset would need to include data so performance can be reported for First Nations patients, and any use would need to demonstrate a collective benefit for that community/communities. Regarding authority to control use of the data, this would sit with the relevant Aboriginal community and their health service (e.g., an Aboriginal Community Controlled Health Organisation), as well as the hospital’s data governance and an ethics committee with appropriate Aboriginal representation.",
    )

    # The dataset is synthetic and has no ethnicity or Indigenous status column. Say what a real setting would add.
    subPrintStep(
        f"\n{STR_BOLD}Step 6 - {STR_RESET}{STR_ITALIC}Where cultural safety issues are not directly evident in the selected dataset, explain what would still need to be considered before applying the system in a real organisational or community setting.{STR_RESET}",
        "Although there are no issues with cultural safety identified in this data, a real hospital would record a patient’s self-identified Indigenous status, enabling the model to be checked for Aboriginal and Torres Strait Islander patients. The organisation would also need to consult with and obtain approval from the local Aboriginal community and their health service on how the data will be collected, stored, and used. Patients and staff must be informed that the model exists, how individuals will be flagged, and who can overturn a patient's flagging, with follow-up actions that are culturally safe (e.g. involving Aboriginal health workers). I would also like to set up a review after the system goes live to identify whether any groups have been under-flagged by the model, and to inform the community and organisation so they can take appropriate action where required.",
    )

    # Section 1, step 1. The purpose of the system. This sits below the row counts above
    # so the numbers it quotes are printed first.
    # Prints this step, in bold italic, into the RTF file. Write your answer between
    # the quotes on the second line. Use triple quotes if it runs over several lines.
    subPrintStep(
        f"\n{STR_BOLD}Step 1 - {STR_RESET}{STR_ITALIC}Define the problem that the AI system addresses and the prediction or decision it is intended to support.{STR_RESET}",
        "Some patients are discharged from the hospital only to return within 30 days. Such readmissions are often stressful for patients and can be costly for hospitals. By predicting readmission among adult inpatients at discharge, the system supports clinicians' decisions about allocating patient follow-up. They can decide whether a patient needs a phone call, a medication review, or an early appointment with the patient’s General Practitioner. A missed readmission is more expensive than a false alarm. The data used for this working example is synthetic and thus should not be used for real-world patients.",
    )

    print(f"\n{STR_BOLD}{STR_UNDERLINE}Criteria for Omission{STR_RESET}")
    print(f"\nRows removed outside of allowed values {dctConfig['data']['allowed_values']}")
    print(f"\nRows removed outside of minimum value ({dctConfig['data']['min_value']})," 
         f" maximum value ({dctConfig['data']['max_value']})\n")

    # Section 1, step 3. Who uses the prediction and who is affected.
    subPrintStep(
        f"\n{STR_BOLD}Step 1 - {STR_RESET}{STR_ITALIC}Explain who might use the prediction and who may be affected by the system.{STR_RESET}",
        "This application is intended for use on adult inpatients. Ingest removes patients outside the age range specified in the config (18 to 100 years of age) and patients whose sex is not on the allowed list in the config (Female or Male). Thus, patients outside these criteria are not covered by this system at all. Direct users of this system are the people planning discharge from hospital of the patients covered by this system: ward doctors, ward nurses, hospital discharge planners and hospital care co-ordinators. In addition, hospital managers and quality assurance personnel will use the totals generated by this application to plan for the follow-up care of their patients. Patients, families and carers of patients will also be affected. As patients who have been flagged for follow-up are passed from hospital to community based primary care and tertiary care, the flagged patients become the workload of GPs and community based nurses.",
    )

    print(f"\n{STR_BOLD}{STR_UNDERLINE}Records Removed from Dataset{STR_RESET}")
    print(f"\nRows removed outside of allowed values {dctConfig['data']['allowed_values']}: {intAfterClean - intAfterValues}")
    print(f"\nRows removed outside of minimum value ({dctConfig['data']['min_value']}), "
         f"maximum value ({dctConfig['data']['max_value']}): {intAfterValues - intCleanRows}\n")

    # The answer to the who-uses question above, widened to everyone the system touches.
    subPrintStep(
        f"\n{STR_BOLD}Step 6 - {STR_RESET}{STR_ITALIC}Identify the users, affected stakeholders and people or communities who could benefit from or be harmed by the system.{STR_RESET}",
        "Patients identified as high risk for readmission can benefit from a flag that prompts follow-up that would otherwise be missed. Clinicians can benefit from a consistent prompt to support their clinical judgement. The hospital can benefit from fewer avoidable readmissions and improved bed management through reduced length of stay. However, some patient groups also face potential harm. Patients the model does not identify as at risk may receive less appropriate care than others with similar needs if clinicians rely on the flag. Similarly, patients who are incorrectly flagged as being at risk and undergo additional contact as a result could experience significant anxiety for no reason. This model does not cover children. The ingest summary for this model shows that 204 rows outside of the age range of (18 to 100) were removed. In this data, these are the under-18s, so the model says nothing about them and must not be applied to them. The summary also shows that 116 rows outside of the allowed values for sex (Female, Male) were removed. In this data, these are patients whose sex is recorded as Other. There were too few of them to score fairly, and the model only knows two categories, so the model says nothing about them either and must not be applied to them. Removing them also means any bias against this group could not be detected. Older patients may also be served least well, because small groups give very unreliable scores. The dataset for this model does not have columns for ethnicity or Indigenous status. Therefore, Aboriginal and Torres Strait Islander peoples and other ethnic communities will not be able to determine whether or not they are receiving different treatment as a result of this model. People in rural and remote areas may also be disadvantaged, as follow-up services and other services required for the model's output may be difficult to access in these locations. Any use of this model would require consultation with the communities of concern and a clear data owner before the model can access patients' health information.",
    )

    # The row counts above are the evidence for what ingestion did.
    subPrintStep(
        f"\n{STR_BOLD}Step 1 - {STR_RESET}{STR_ITALIC}Pipeline architecture diagram: describe data ingestion.{STR_RESET}",
        "The stage of ingestion is ingest.py. This is the only stage that actually reads the raw file. It reads a file specified in the config (e.g. a local file or a web page) and keeps only the columns listed in the config. It removes duplicates, then rows that are outside of the sex specified in the config and age outside of 18-100. It holds out 5% of the remaining patients as “new patients” and writes their answers to a file; it then writes the rest of the cleaned data to a file that the remaining stages read from. For each run of the stages, it prints where the data came from, when it was loaded, and a “fingerprint” of the file, so later stages and a human can tell where the data came from.",
    )

    # The cleaning and filtering above, and the counts printed, are the evidence for how bad rows are handled.
    subPrintStep(
        f"\n{STR_BOLD}Step 2 - {STR_RESET}{STR_ITALIC}Describe how invalid or incomplete data is handled.{STR_RESET}",
        "The Ingest stage of the pipeline removes invalid rows and reports how many. Spaces are stripped from string columns, duplicates are removed, and rows are removed where the sex is not Female or Male (116) and outside the age range of 18 to 100 (204), and the counts are printed above. The raw data had no missing values. The pipeline never fills in gaps or ‘guesses’ values in any of the stages. The Validate stage checks the file after cleaning and stops the pipeline if there are any problems with the cleaned data. The Validate stage stops at once if the table is empty or a required column is missing. Otherwise it runs the remaining checks together and reports every problem in one message: too few rows, missing values in a required column, a value outside the allowed list for sex, an age outside the range, duplicate rows, and a target that is not 0 or 1. The feature stage also refuses any category that is not on the config list, so bad data is stopped as early as possible before it causes problems later in the pipeline.",
    )

    # The cleaned profile above shows what the data does and doesn't hold. Think about synthetic values, clipped ranges and 0 ages.
    subPrintStep(
        f"\n{STR_BOLD}Step 6 - {STR_RESET}{STR_ITALIC}Discuss limitations in the data, features, model and evaluation (the data).{STR_RESET}",
        "There are no date or site columns, so data cannot be split into training and test data for different times or sites. A random split of the data will likely overstate how well the model works in practice. There are no columns for Indigenous status or ethnicity, so fairness can only be assessed by sex and age. The average readmission rate is about 14%. Due to this strongly imbalanced class distribution, the results must be interpreted with caution.",
    )


def subAnswerValidate(dctConfig: dict, strInputPath: str) -> None:
    """Print the validate answers, each below its evidence.

    Args:
        dctConfig: The loaded config.yaml settings.
        strInputPath: The cleaned CSV to check.

    Returns:
        None. It only prints.
    """
    print(f"\n{STR_BOLD}{STR_UNDERLINE}ANSWERS: VALIDATE{STR_RESET}")

    # Run the same checks as validate.py. They print PASS or FAIL for each one.
    dfData = dfLoadCsv(strInputPath)
    dfValidate(dfData, dctConfig)
    print(f"{len(dfData)} rows passed validation")

    # Passing validation above is the evidence. List the rules this stage checks.
    # Prints this step, in bold italic, into the RTF file. Write your answer between
    # the quotes on the second line. Use triple quotes if it runs over several lines.
    subPrintStep(
        "Step 1 - Pipeline architecture diagram: describe data validation.",
        "Step 2 - Add checks for relevant data-quality problems, such as: missing or unexpected columns; incorrect data types; missing values; invalid value ranges; duplicate rows; categories outside an expected set; date or time values that cannot be parsed.",
    )

    # The rules in dfValidate ran above. Say which of these problems each rule covers and which it does not.
    subPrintStep(
        "Step 2 - Add checks for relevant data-quality problems, such as: missing or unexpected columns; incorrect data types; missing values; invalid value ranges; duplicate rows; categories outside an expected set; date or time values that cannot be parsed.",
        "The checks test for unexpected columns, missing and wrong data types, missing values, out of range values, duplicate rows and values that are not in the allowed categories. Check 2 fails the run if a feature column or the target column is missing. Check 9 fails any column that ingest does not keep. Check 10 checks if all numeric feature columns only hold numbers. Check 4 counts missing values. Check 6 checks the values for the column “age” against 18 to 100. Check 11 checks the values for the columns “BMI”, “blood_pressure”, “glucose”, “cholesterol” and “creatinine” against the limits in the config file. Check 5 tests the values for the column “sex” against the allowed list of values. Check 7 counts the number of duplicate rows. Check 8 fails if the target column is not 0 or 1. There are no date or time columns in the dataset.",
    )

    # Validation is one automated check. The pytest tests and the CI workflow are the others.
    subPrintStep(
        "Step 5 - Add automated quality checks through tests, validation scripts, continuous integration workflows or another justified approach.",
        "Automated quality control is applied in two ways: (1) the validate stage of the cleaning process itself runs 11 tests on the cleaned data, which are reported as PASS/FAIL above; (2) the train/evaluate/comparison/plots all call the same 11 tests (which are fail-fast), thus none of them can be run on data that fails to pass the earlier tests. A pytest test suite of 175 tests for the cleaning and filtering, and for loading a file or a web page, as well as for the config checks, the data validation, the feature generation and the RTF reports, runs with its own minimal config file, so be careful when editing config.yaml as you may break the tests. The tests are also run on push and pull requests to GitHub as part of a GitHub Actions workflow.",
    )

    # Describe .github/workflows/ci.yml: when it runs and what it runs.
    subPrintStep(
        "Step 5 - Where appropriate, configure a workflow such as GitHub Actions to run selected checks automatically.",
        "A GitHub Actions workflow (.github/workflows/ci.yml) runs all the tests automatically on a clean GitHub-hosted machine (separate from my dev box). The workflow installs all required libraries from requirements.txt and then runs python -m pytest tests. All failing test cases of the executed test suite will then turn the complete run red on the Actions tab. Note that this workflow only runs tests and does not run the full pipeline. DVC manages the raw data, and thus it is not stored in Git.",
    )

    # The validation rules above stop the run when the data changes shape. Say what happens next and who is told.
    subPrintStep(
        "Step 6 - Identify operational risks, such as changes to data formats and failed validation checks.",
        "If the supplier changes the data format (i.e. renaming or removing columns), it will fail validation check 2, and the pipeline stops. An extra column will fail validation check 9. Text in a numeric column or an impossible value will fail on validation checks 10 and 11. All of these problems will surface in one message. Nothing will alert a person that a run has failed, so they will have to read the run report to determine this. The limits in the config.yaml also carry risk. If the limits are set too high or low, it could allow bad data through. Slow drift in values within the defined limits will not fail any checks.",
    )


def subAnswerTrain(dctConfig: dict, strInputPath: str) -> None:
    """Print the train answers, each below its evidence.

    Args:
        dctConfig: The loaded config.yaml settings.
        strInputPath: The cleaned CSV the model was trained from.

    Returns:
        None. It only prints.
    """
    print(f"\n{STR_BOLD}{STR_UNDERLINE}ANSWERS: TRAIN{STR_RESET}")

    # The same features and split that train.py used, rebuilt in memory.
    dfData = dfLoadCsv(strInputPath)
    dfX, serY = tplBuildFeatures(dfData, dctConfig)
    dfXTrain, dfXTest, serYTrain, serYTest = lstSplitData(dfX, serY, dctConfig)
    pthOut = Path(dctConfig["model"]["path"])
    pthCard = Path(dctConfig["approval"]["model_card_path"])

    # The target and feature table built above are the evidence for the target and the unit.
    # Prints this step, in bold italic, into the RTF file. Write your answer between
    # the quotes on the second line. Use triple quotes if it runs over several lines.
    print(f"\n{STR_BOLD}Target Variable:{STR_RESET} {dctConfig['features']['target_column']}")

    subPrintStep(
        "Step 1 - Identify the target variable, whether the task is classification or regression, and the unit for which each prediction is made.",
        "The target is readmission_30d. If the value is 1, the patient was readmitted within 30 days of discharge and 0 if not. This is a binary classification, not a regression. Each prediction is made for one adult inpatient at the point of discharge. The model gives a probability of readmission for that patient, and a flag is raised when it is high enough.",
    )

    # Section 1, step 4. The prediction time decides which inputs are allowed.
    subPrintStep(
        "Step 1 - Define the prediction time: the point at which the prediction would be generated in a real workflow.",
        "This prediction is made at the time of discharge from hospital, after the clinical team have decided that it is safe for the patient to go home. The model uses data that has already been recorded for the patient: age, sex, BMI, blood pressure, glucose, total cholesterol, creatinine, presence of diabetes and presence of hypertension. A simple flag is then generated to assist clinical staff with any further follow-up required before the patient leaves hospital.",
    )

    # dfX holds the inputs the model sees. Check each one is known at the prediction time.
    subPrintStep(
        "Step 1 - Identify which inputs would genuinely be available at the prediction time. Information that becomes available only afterwards cannot legitimately be used as a feature.",
        "The following 10 input variables are measured before a patient’s discharge and can be used for prediction: age, sex, BMI, systolic and diastolic blood pressure, glucose, cholesterol, creatinine, diabetes, and hypertension. Note that the mortality variable is not used in the model, as it is measured only after discharge and would otherwise leak part of the outcome into the data used for prediction. The Patient ID variable is a label and is not used as an input variable; it is also not included in the cleaned data file, whereas the diagnosis variable is included in the cleaned data file but not used as an input variable in the model. All columns that are excluded from the model are listed in the config file and in the config check to prevent them from being added as input variables.",
    )

    # dfEngineerFeatures built the extra columns inside tplBuildFeatures above.
    subPrintStep(
        "Step 1 - Pipeline architecture diagram: describe feature engineering and preprocessing.",
        "Feature engineering sits in features.py. The function creates two features. The first is pulse_pressure, the difference between systolic and diastolic blood pressure. The second is sex_Male, which converts sex to a 0/1 feature where males are 1 and females are 0. The config, not the data, determines the sex categories. Therefore, any values not in the category list will end the run rather than be scored against the first category. The function does not learn anything from the data (i.e., no scaling, no averaging, etc.), so it is safe to use on the training data, the test data, and new patients. ",
    )

    # dfEngineerFeatures built pulse_pressure and sex_Male inside tplBuildFeatures above.
    subPrintStep(
        "Step 2 - Create at least two meaningful engineered or transformed features and explain their intended role.",
        "The dfEngineerFeatures builds two features. The first is pulse_pressure, the difference between systolic and diastolic blood pressure. This feature captures some cardiovascular stress not captured by the two individual blood pressure readings. The second feature is sex_Male, a text feature that is converted to a numeric feature (1 for male and 0 for female) in the config. Since logistic regression only uses numeric features, female is the reference value for this feature. This feature is not learned from the data and is set in the config.",
    )

    # The same feature function runs in training and in predict.py, and the stages run from one config.yaml.
    subPrintStep(
        "Step 2 - Implement preprocessing through a repeatable pipeline or clearly organised set of scripts.",
        "Ingest, validate, and feature preprocessing are each split into a single named script: ingest.py, validate.py and features.py. Each script can be run with a single command such as python -m src.ingest. The steps are chained together in dvc.yaml so that dvc repro will run them in the correct order after any changes. All configuration settings are in config.yaml. Each step is run in exactly the same way every time.",
    )

    subPrintDataSplitEvidence(dctConfig, dfXTrain, dfXTest)

    # The split happened above before any model was fitted. Say which transformations are fitted and which are fixed rules.
    subPrintStep(
        "Step 2 - Ensure that transformations are fitted using training data only where this is required to avoid leakage.",
        "None of the transformations in this pipeline learn anything from the data. The two engineered features are fixed rules set in the config (a simple subtraction, and a 0/1 category column). There’s no scaling and no gap filling, so nothing can leak from the test data through this part of the pipeline. The only part that learns anything is the model itself, and it’s trained on the training rows only. Cross-validation also uses the training rows only, and the test data is scored once, at the end. If scaling or imputation were added later, it would need to be a scikit-learn transform within a pipeline, and fitted on the training data only.",
    )

    # See excluded_columns in config.yaml and the shared dfEngineerFeatures. Pick the risk you want to explain.
    subPrintStep(
        "Step 2 - Identify at least one target-leakage, future-information or train-serve consistency risk and explain how it is controlled.",
        "The main leakage risk is mortality. It’s a post-discharge measurement, so using it as an input would hand the model the outcome. It’s listed in excluded_columns in the config for this reason, and the config check would prevent the config from being loaded if an excluded column was also in feature_columns. A further risk here is train-serve mismatch, where patients are prepared differently for new predictions than they were for the data used in training. All uses of the data (training, evaluation, prediction and web service) use a single shared function, dfEngineerFeatures, to prepare the features. This function also checks that all categories are in the config list and stops the run if not.",
    )

    # The features built above are all the model sees. Say what is missing and what is only a proxy.
    subPrintStep(
        "Step 6 - Discuss limitations in the data, features, model and evaluation (the features).",
        "The input file contains 10 single-time-point clinical measures. This dataset omits many well-known readmission risk predictors, such as the number of previous admissions, length of stay, medications, and support at home. Diabetes and hypertension are included as simple binary flags, but no indication of severity or degree of control of either of these conditions is provided. The two engineered features, pulse_pressure and sex_Male, are basic. Sex is restricted to male or female. There are no ethnicity, Indigenous status or location columns, so the model can’t be checked for whether it treats different groups differently. The data is synthetic, so the model can only learn the relationships embedded in the data generator.",
    )

    subPrintTrainingEvidence(dctConfig)

    # The training line above is the evidence for model training.
    subPrintStep(
        "Step 1 - Pipeline architecture diagram: describe model training.",
        "The training stage (train.py), loads and validates the cleaned data, builds the features, splits off the test rows and fits a logistic regression on the 7,356 training rows, using the settings in the model section of the config.yaml file. Before anything can be saved, it must first pass through an approval gate. The cross-validated AUC on the training rows needs to be at least 0.050 above the baseline AUC. The cross-validation AUC for the logistic regression was 0.606 or 0.106 above the baseline value of 0.500. If the gate test fails, the model does not save it. The test ends with an error. A model that passes the test is saved as model.joblib along with a model card (model_card.json) and the run is logged in MLflow.",
    )

    # The model file above is an artefact. Say how it and the data are versioned (dvc.yaml).
    subPrintStep(
        "Step 1 - Pipeline architecture diagram: describe versioning and experiment or artefact tracking.",
        "Git versions the code, config.yaml, dvc.yaml and requirements.txt. DVC versions the raw data using a small pointer file and tracks the outputs of each stage (model, model card, metrics and reports) by storing their hashes in dvc.lock. MLflow tracks the runs for training, evaluation and comparison with relevant parameters, scores and saved files. Every run is tagged with the Git commit and fingerprints of the data and config. The model card carries the same fingerprints, so you can see what was used to build a given model.",
    )

    print(f"\n{STR_BOLD}Required gain:{STR_RESET} {dctConfig['approval']['min_auc_gain']:.3f}")
    # The saved model file above is the evidence. Say whether the model is approved before use.
    subPrintStep(
        "Step 1 - Pipeline architecture diagram: describe model saving or approval.",
        "A model is only saved if it passes the approval gate in train.py. The cross-validated AUC on the training rows needs to be at least 0.050 above the baseline AUC. A model that fails isn’t saved and the stage stops with an error. A model that passes the test is saved as model.joblib along with a model card (model_card.json). Additionally, a named person must still approve it before any real use. Passing the gate is an automated check, not clinical approval.",
    )

    print(f"\n{STR_BOLD}{STR_UNDERLINE}Git history{STR_RESET}")
    try:
        objLog = subprocess.run(["git", "log", "--date=short", "--format=%h  %ad  %s"],
                                capture_output=True, text=True, encoding="utf-8", timeout=10)
        strLog = objLog.stdout.strip() if objLog.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        strLog = ""

    print(strLog or "No Git history found. Git isn't installed, or this folder isn't a Git repository.")

    # No code produces this evidence. It lives in the Git history, so describe your commits here.
    subPrintStep(
        "Step 4 - Use Git throughout the project and make meaningful commits that reflect the development of the pipeline.",
        "Git was used to commit the pipeline in the order it runs, and the history is pushed to a private GitHub repository. Five commits build out the stages of the pipeline in sequence, ending with the tests and CI workflow. Four more initialise DVC, track the raw data with a pointer file, and record the first run by committing dvc.lock. The commit message for each one sums up what was added.",
    )

    # print the tracking evidence
    subPrintVersionEvidence()

    # config.yaml, requirements.txt and dvc.yaml are the evidence. Say what each one pins.
    subPrintStep(
        "Step 4 - Track the code, configuration and environment needed to reproduce the project.",
        "All the code is tracked in Git. The stage scripts are in src/, and the tests are in tests/. All the configuration is stored in a file config.yaml. This file contains all code settings, including the random seed for random number generation. The dvc.yaml file details all the stages. For each stage, it lists all inputs and outputs. The hashed values of the inputs and outputs for a given run are stored in dvc.lock. The environment is defined in requirements.txt. In the GitHub Actions workflow, the Python version is set to 3.12. Most libraries in requirements.txt only have a minimum version, so a later install could pick up newer releases and give slightly different results.",
    )

    subPrintModelProvenanceEvidence(pthCard, pthOut, strInputPath)

    # The model file above came from the data, code and config at this commit. Explain how to trace it (dvc.lock, Git commit).
    subPrintStep(
        "Step 4 - Explain how another team member could identify which data, code, configuration and evidence produced the selected model.",
        "The model card saved beside the model records the Git commit, the random seed, and fingerprints of the model, the data and config.yaml. The commit provides the same code and config from Git, and dvc.lock, with the raw data’s pointer file, lets DVC restore the matching data. The same commit and fingerprints are tagged on the MLflow run, which also holds the parameters and scores. As long as the restored files match the fingerprints on the card, the team member will have the same model, data and configuration.",
    )

    # The model file saved above is reused by evaluate.py and predict.py. Say what it contains and what it does not.
    subPrintStep(
        "Step 5 - Save the trained preprocessing and modelling pipeline in a form that can be reused.",
        "The fitted pipeline is saved as a single file, artifacts/model.joblib. It holds two steps: the feature step (dfEngineerFeatures) and the logistic regression. It can score raw patient data on its own. evaluate.py, predict.py, plots.py and the web service load it without retraining.",
    )

    # The saved model above is only a file. Say who must approve it, and what checks, before anyone uses it.
    subPrintStep(
        "Step 6 - Identify appropriate human oversight, governance and approval requirements before real-world use.",
        "Before any real use, a named person would need to approve it. This solution must be tested on real hospital data, checked for fairness across sex and age bands, and signed off by individuals responsible for clinical safety and privacy. The model would flag patients who need extra monitoring, but a clinician would make the decision and could override the model’s recommendations. The owner of the model would be responsible for monitoring its performance and retraining or withdrawing it as necessary. Before real use, approval would also need to follow the CARE principles, so Aboriginal and Torres Strait Islander people would have authority over how their data is used and the model would need to show collective benefit for their communities.",
    )


def subAnswerEvaluate(dctConfig: dict, strInputPath: str) -> None:
    """Print the evaluate answers, each below its evidence.

    Args:
        dctConfig: The loaded config.yaml settings.
        strInputPath: The cleaned CSV the model was tested on.

    Returns:
        None. It only prints.
    """
    print(f"\n{STR_BOLD}{STR_UNDERLINE}ANSWERS: EVALUATE{STR_RESET}")

    # The same split as evaluate.py, then the saved model scored on the test rows.
    # Nothing is written, so metrics.json is left as the evaluate stage made it.
    dfData = dfLoadCsv(strInputPath)
    dfX, serY = tplBuildFeatures(dfData, dctConfig)
    _, dfXTest, serYTrain, serYTest = lstSplitData(dfX, serY, dctConfig)
    mdlModel = mdlLoadModel(dctConfig["model"]["path"])
    intPositive = list(mdlModel.classes_).index(dctConfig["evaluation"]["positive_label"])
    dfRawTest = dfRawInputs(dfData, dctConfig).loc[dfXTest.index]
    arrProbabilities = mdlModel.predict_proba(dfRawTest)[:, intPositive]
    arrPredictions = mdlModel.predict(dfRawTest)
    dctMetrics = dctComputeMetrics(serYTest, arrPredictions, dctConfig, arrProbabilities)

    subPrintEvaluationStrategyEvidence(dctConfig, serYTrain, serYTest)

    # The stratified split rebuilt above is the evaluation strategy. Justify the split, the seed and cross-validation.
    subPrintStep(
        "Step 3 - Select and justify a suitable train-validation-test or other evaluation strategy.",
        "Five percent of the cleaned patients are set aside at ingest as new patients, and the rest are split 80/20 into training and test rows. There is no separate validation set. For training rows five-fold cross-validation is used to compare models and drive the approval gate, while test rows are scored once after that. The split is stratified. This means the resulting split has readmission rates of about 14% in both the training and test sets. The split also uses the seed in config.yaml at every stage. A random split was the only option, since there is no date or site column, so results may look better than they would on a later period or a different hospital.",
    )

    print(f"\n{STR_BOLD}{STR_UNDERLINE}Metrics Information{STR_RESET}")
    print(f"\n{STR_BOLD}Metrics results:{STR_RESET} {dctMetrics}")

    # The metrics above are the evidence for how the saved model performs on unseen rows.
    # Prints this step, in bold italic, into the RTF file. Write your answer between
    # the quotes on the second line. Use triple quotes if it runs over several lines.
    subPrintStep(
        "Step 1 - Pipeline architecture diagram: describe model evaluation and benchmarking (the evaluation stage).",
        "The saved pipeline is then loaded by evaluate.py and used to score the isolated test rows. The rows used for testing here were not seen by train.py. Accuracy, F1, precision, recall, and ROC AUC are written to metrics.json, error counts and scores by sex and age band are printed, and the run is logged in MLflow with the model’s fingerprint. Benchmarking is done by compare.py on the same rows, which scores the saved model’s type and a random forest against the benchmarks defined in config.yaml, a do-nothing baseline and a simple rule. The results are documented in model_comparison.csv and evaluation_evidence.md.",
    )

    dfGroups = dfBuildGroups(dfData, dfXTest, dctConfig)
    if len(dfGroups.columns) > 0:
        dfByGroup = dfMetricsByGroup(dfGroups, serYTest, arrPredictions, arrProbabilities, dctConfig)
        print(f"\n{STR_BOLD}{STR_UNDERLINE}Scores By Group{STR_RESET}")
        print(dfByGroup.round(3).to_string(index=False))
        subPrintGroupErrorEvidence(dfByGroup, dctConfig)

    # The scores-by-group table above is the evidence. Read the recall, precision and flagged rate for each group.
    subPrintStep(
        "Step 6 - Consider whether errors may be distributed unevenly across relevant groups.",
        "Yes, errors are spread a little unevenly. The model catches fewer readmissions among men than women (recall 0.593 against 0.637) and ranks them less well (ROC AUC 0.595 against 0.679). The lowest recall is in the 80 to 120 age band (0.480, against 0.603 to 0.667 in the other bands), but that band has only 155 test rows, so the figure is unstable. The model flags a similar share of every group (36 to 40 percent), so the gaps come from how many of the real readmissions it identifies. The only two values that can be checked for this data are sex and age band as there are no columns for ethnicity or Indigenous status.",
    )

    dfCounts = dfConfusionCounts(serYTest, arrPredictions, dctConfig)
    print(f"\n{STR_BOLD}{STR_UNDERLINE}Error Counts On The Test Rows{STR_RESET}")
    print(dfCounts.to_string(index=False))
    subPrintConsequencesEvidence(dfCounts)

    # The error counts above give the numbers. Say what a missed readmission costs and what a false alarm costs.
    subPrintStep(
        "Step 6 - Explain the consequences of false positives, false negatives or other important model failures in the context of the selected task.",
        "A false negative is a readmitted patient the model missed, so the patient leaves without the extra monitoring that might have prevented their return. The model missed 99 of the 258 readmissions in the test rows. A false positive is a flagged patient who was not readmitted, which means unnecessary monitoring and clinician time. There were 552 of these, so about 78 percent of flagged patients were false alarms. Assuming extra follow-up is of low risk, we trade off recall against precision. However, many false alarms would likely swamp a clinical team, and we would not wish a model to override a clinician’s decision given the weak signal in this case.",
    )

    subPrintMetricChoiceEvidence(dctConfig, serY, serYTest, dctMetrics)

    # The metrics above are the evidence. Say why recall, precision and AUC suit a rare outcome.
    subPrintStep(
        "Step 3 - Evaluate the models using at least one metric suited to the prediction task.",
        "The primary metric of interest for this problem is ROC AUC, since our goal is to rank patients from most likely to be readmitted to least likely to be readmitted. Recall and precision are reported, since both a failure to predict a readmission and a false alarm are costly. Accuracy is reported but can be misleading, as only 14% of patients are readmitted. The implication is that a model that predicts no readmission for everyone would be correct 86% of the time. On the test set the logistic regression achieves an AUC of 0.639, with a recall of 0.616 and a precision of 0.224. The do-nothing baseline achieves an AUC of 0.500.",
    )

    subPrintRunHistoryEvidence(dctConfig)

    # The metrics file written above sits in the reports folder. Justify that choice.
    subPrintStep(
        "Step 4 - Use MLflow, a structured metrics and artefact folder, or another justified approach to record model runs and outputs.",
        "Runs are logged locally to a database file (mlflow.db). The report files are overwritten on each run, so MLflow keeps the history. The train, evaluate and compare stages log their settings, scores and output files. Each run is tagged with the Git commit ID and the fingerprints of the data, config and model it used. Outputs go to fixed folders: model.joblib and model_card.json in artifacts, and the reports in reports. The database isn’t committed to Git, because the history belongs to the machine that generated it.",
    )

    # Say where each of these is recorded. The metrics are above. The rest is in config.yaml, dvc.lock and the model file.
    subPrintStep(
        "Step 4 - Record relevant information such as: model and preprocessing versions; parameters; evaluation metrics; data or dataset version; configuration; saved model artefacts; experiment results.",
        "Each train, evaluate and compare run is recorded in MLflow with its parameters (model settings, feature rules, test size and random seed), its scores, and tags for the Git commit and the fingerprints of the raw data, cleaned data, config and model. The code and preprocessing version are recorded by the Git commit, as the feature step is saved in the model file as a reference to the code, so that commit has to be checked out to rebuild the same behaviour. Config.yaml is stored with every run, so the exact settings are saved with the results. The saved model and model card (model.joblib and model_card.json, which holds the model, data and config fingerprints) are saved in artifacts/ and copied to the training run, and the scores and comparison results are in reports/ and on the evaluate and compare runs. Library versions are printed in the train report but not stored in MLflow, and most are pinned only to a minimum.",
    )


def subAnswerCompare(dctConfig: dict) -> None:
    """Print the compare answers, each below its evidence.

    Args:
        dctConfig: The loaded config.yaml settings.

    Returns:
        None. It only prints.
    """
    print(f"\n{STR_BOLD}{STR_UNDERLINE}ANSWERS: COMPARE{STR_RESET}")

    # The table compare.py wrote, read back from disk.
    subPrintComparisonEvidence(dctConfig)

    # The table above is the evidence. Say what the baseline and candidates show.
    # Prints this step, in bold italic, into the RTF file. Write your answer between
    # the quotes on the second line. Use triple quotes if it runs over several lines.
    subPrintStep(
        "Step 1 - Pipeline architecture diagram: describe model evaluation and benchmarking (the baseline and candidate comparison).",
        "",
    )

    # The whole pipeline has run by now. Describe the architecture diagram (the image goes in the README).
    subPrintStep(
        "Step 1 - Create a pipeline architecture diagram showing: raw data and its source, data ingestion, data validation, feature engineering and preprocessing, model training, model evaluation and benchmarking, versioning and experiment or artefact tracking, model saving or approval, the reusable prediction workflow, and monitoring considerations.",
        "",
    )

    # Tie the answer to this task: rare readmissions, a shared feature function, config-held exclusions.
    subPrintStep(
        "Step 1 - Explain why the proposed architecture is suitable for your prediction task rather than presenting a generic machine learning pipeline.",
        "",
    )

    # The baseline row in the table above is a dummy classifier that always predicts the commoner outcome.
    subPrintStep(
        "Step 3 - Train a simple baseline model, such as a dummy classifier, dummy regressor, historical average, persistence model or existing-rule benchmark.",
        "",
    )

    # The logistic regression and random forest rows in the table above are the candidates.
    subPrintStep(
        "Step 3 - Train at least one candidate machine learning model.",
        "",
    )

    # Every model was given the same engineered features and the same split, shown above.
    subPrintStep(
        "Step 3 - Apply preprocessing consistently to the baseline and candidate models where appropriate.",
        "",
    )

    # Read the baseline and candidate rows of the table above side by side.
    subPrintStep(
        "Step 3 - Compare the candidate model with the baseline.",
        "",
    )

    # Say what the numbers mean for readmission prediction, not just what they are.
    subPrintStep(
        "Step 3 - Interpret the results rather than reporting metric values alone.",
        "",
    )

    # Think about synthetic data, one split, the threshold and what AUC does not tell you.
    subPrintStep(
        "Step 3 - Discuss limitations of the evaluation, including what the results do and do not establish.",
        "",
    )

    # The table above is from one split of synthetic data. Say what it does and does not establish.
    subPrintStep(
        "Step 6 - Discuss limitations in the data, features, model and evaluation (the model and the evaluation).",
        "",
    )

    # The README and Git guide hold the steps. Summarise them here.
    subPrintStep(
        "Step 4 - Provide sufficient set-up and run instructions for another person to reproduce the result.",
        "",
    )

    # Each stage is a python -m command, and dvc.yaml runs them all with dvc repro.
    subPrintStep(
        "Step 5 - Provide repeatable commands or scripts for running the main stages of the pipeline.",
        "",
    )


def subAnswerPredict(dctConfig: dict) -> None:
    """Print the predict answers, each below its evidence.

    Args:
        dctConfig: The loaded config.yaml settings. Uses outputs.predictions_path.

    Returns:
        None. It only prints.
    """
    print(f"\n{STR_BOLD}{STR_UNDERLINE}ANSWERS: PREDICT{STR_RESET}")

    # The predictions file python -m src.predict wrote.
    pthPredictions = Path(dctConfig["outputs"]["predictions_path"])
    if pthPredictions.is_file():
        print(f"\n{STR_BOLD}Predictions file:{STR_RESET} {pthPredictions}, {len(pd.read_csv(pthPredictions))} rows")
    else:
        print(f"\nNo predictions at {pthPredictions}. Run python -m src.predict first.")

    # The predictions above came from the saved model and the shared feature function.
    # Prints this step, in bold italic, into the RTF file. Write your answer between
    # the quotes on the second line. Use triple quotes if it runs over several lines.
    subPrintStep(
        "Step 1 - Pipeline architecture diagram: describe the reusable prediction workflow.",
        "",
    )

    # What would you watch once this runs for real: input drift, the readmission rate, recall over time.
    subPrintStep(
        "Step 1 - Pipeline architecture diagram: describe monitoring considerations.",
        "",
    )

    # Think about input drift, the readmission rate, recall over time and how often validation fails.
    subPrintStep(
        "Step 6 - Explain what should be monitored if the system were used beyond the project.",
        "",
    )

    # python -m src.predict wrote the predictions above for the new data file.
    subPrintStep(
        "Step 5 - Provide a script, command or clearly identified notebook section that can generate predictions for new data.",
        "",
    )

    # predict.py calls the same dfEngineerFeatures as training. Show it, for example with a test or a worked row.
    subPrintStep(
        "Step 5 - Demonstrate that the prediction workflow applies the same preprocessing and feature logic used during training.",
        "",
    )

    # Think about serving, monitoring, retraining, approval and data access.
    subPrintStep(
        "Step 5 - Explain what would be needed to extend the workflow into a production deployment.",
        "",
    )

    # The predictions above came from one batch. Say what you would check on each new batch before trusting it.
    subPrintStep(
        "Step 6 - Identify operational risks, such as missing inputs, model or data drift, and predictions outside expected ranges.",
        "",
    )


def subAnswerPlots(dctConfig: dict) -> None:
    """Print the plots answers, each below its evidence.

    Args:
        dctConfig: The loaded config.yaml settings.

    Returns:
        None. It only prints.
    """
    print(f"\n{STR_BOLD}{STR_UNDERLINE}ANSWERS: PLOTS{STR_RESET}")

    # The graph files plots.py wrote.
    subPrintGraphEvidence(dctConfig)

    # The graphs are the evidence. Say what each one shows and what it means for use.
    subPrintStep(
        "Step 3 - Evaluate the models using at least one metric suited to the prediction task.",
        "All the graphs in this evaluation use the same test data. The ROC curves have AUC values for the saved logistic regression of 0.639, the random forest of 0.636 and the age rule of 0.511 where 0.5 is chance. So all models perform only slightly better than chance. The precision-recall curves for rare events look rather poor and only show marginal improvement over the 14% readmission rate, which is the precision of a model that always flags all patients (and hence has 100% recall). The error grid for the saved logistic regression is shown in the figure below. It reports 159 readmissions found, 99 missed and 552 false alarms.",
    )


def subMain() -> None:
    """Command line entry point: print every answer with the evidence behind it."""
    objParser = argparse.ArgumentParser(description="Print the assessment answers with their evidence.")
    subAddConfigArgument(objParser)
    objParser.add_argument("--input", default=None, help="CSV to use (default: data.processed_path)")
    objArgs = objParser.parse_args()

    dctConfig = dctLoadConfig(objArgs.config)
    strInputPath = objArgs.input or dctConfig["data"]["processed_path"]

    # Everything printed inside this block goes to the RTF file, not the screen.
    with objCaptureToRtf(dctConfig["outputs"]["rtf_path"], "Answer_Evidence"):
        # Display the file header at the top of the stage output
        subPrintPiplineHeader(
            "Answer_Evidence",
            dctConfig["project"]["client_question"],
            dctConfig["features"]["target_column"],
            ", ".join(dctConfig["features"]["feature_columns"]),
            strInputPath,
        )

        subAnswerIngest(dctConfig)
        subAnswerValidate(dctConfig, strInputPath)
        subAnswerTrain(dctConfig, strInputPath)
        subAnswerEvaluate(dctConfig, strInputPath)
        subAnswerCompare(dctConfig)
        subAnswerPredict(dctConfig)
        subAnswerPlots(dctConfig)


if __name__ == "__main__":
    subMain()
