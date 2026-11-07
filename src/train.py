"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    train.py
    
    Stage 4 of the pipeline: model training.
    
    Loads and validates the data, builds features, fits a model on the training
    split, and saves the fitted model so that evaluate.py and predict.py can load
    it later without retraining.
    
    Before saving, the model has to pass an approval gate: its cross-validated AUC
    must beat the do-nothing baseline by at least approval.min_auc_gain. A model
    that fails is not saved and the stage stops with ModelGateError. A model that
    passes is saved with a model card (JSON) that says what it is for, what built
    it and the gate result. Passing the gate is not clinical approval. A person
    still has to sign off.

    This file also holds pthSaveModel() and mdlLoadModel(), which the other stages reuse,
    and mdlBuildModel(), which compare.py uses to build the baseline, the rule
    benchmark and the candidates.
    
    The model settings and the save location come from config.yaml (the model section).
    
    The printed output is saved to an RTF file instead of the screen. The file name
    comes from outputs.rtf_path in config.yaml (the text {stage} becomes the stage name).

    Run from the project root:
        python -m src.train
"""

import argparse
import datetime
import json
import subprocess
import platform
import yaml

# joblib saves and loads Python objects (like a fitted model) to and from disk.
# It is the standard choice for scikit-learn models.
import joblib

# pandas joins the training and test tables into one file.
import numpy as np
import pandas as pd
import importlib.metadata as objMeta

# The class for working with file and folder paths
from pathlib import Path

# The three kinds of model the pipeline can build: a do-nothing baseline, a
# logistic regression (simple and fast) and a random forest (many decision trees).
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import subAddConfigArgument, dctLoadConfig
# Saves what the stage prints to an RTF file instead of the screen.
from src.display import (objCaptureToRtf, bDisplayDatasetDetails, subPrintStep, subPrintPiplineHeader,
                         STR_BOLD, STR_UNDERLINE, STR_RESET, STR_ITALIC)
# Records the run (settings, files, tags) in MLflow when tracking is on.
from src.tracking import bLogRun, strFileFingerprint, strGitCommit
from src.features import tplBuildFeatures, lstSplitData, objBuildFeatureStep, dfRawInputs, dfEngineerFeatures
from src.ingest import dfLoadCsv
from src.validate import dfValidate


class ModelGateError(Exception):
    """Raised when a trained model fails the approval gate, so it must not be saved."""


class RuleBenchmark(ClassifierMixin, BaseEstimator):
    """A fixed clinical-style rule that behaves like a model, for benchmarking.

    It flags a patient as positive when one input column is at or above a
    threshold, for example age at or above 65. It learns nothing from the data.
    A candidate model should beat a rule this simple before it earns a place.

    The argument names (column, threshold, positive_label) have no prefixes
    because scikit-learn reads them by name when it copies the estimator.
    """

    def __init__(self, column="age", threshold=65, positive_label=1):
        self.column = column
        self.threshold = threshold
        self.positive_label = positive_label

    def fit(self, dfX, serY):
        """Record the target values the rule can return. Nothing is learned.

        Args:
            dfX: Training inputs. Must contain the rule's column.
            serY: Training targets.

        Returns:
            This object, so calls can be chained.

        Raises:
            ValueError: If the rule's column is not in the inputs.
        """
        if self.column not in dfX.columns:
            raise ValueError(f"Rule benchmark needs column '{self.column}', which is not in the model inputs.")
        self.classes_ = np.array(sorted(set(serY)))
        return self

    def predict_proba(self, dfX):
        """Return a probability pair per row: 1 for the flagged class when the rule fires.

        Args:
            dfX: Inputs to score.

        Returns:
            An array with one column per class, in classes_ order. The rule
            gives only 0 or 1 as its score.
        """
        arrFires = (dfX[self.column].to_numpy() >= self.threshold).astype(float)
        intPositive = list(self.classes_).index(self.positive_label)
        arrProbabilities = np.zeros((len(dfX), len(self.classes_)))
        arrProbabilities[:, intPositive] = arrFires
        arrProbabilities[:, 1 - intPositive] = 1 - arrFires
        return arrProbabilities

    def predict(self, dfX):
        """Return the class the rule picks for each row.

        Args:
            dfX: Inputs to score.

        Returns:
            An array of class values: the positive label where the rule fires.
        """
        return self.classes_[self.predict_proba(dfX).argmax(axis=1)]


def mdlBuildModel(dctSettings: dict, intSeed: int):
    """Build an unfitted model from a settings dictionary.

    One place builds every model, so the training stage and the comparison stage
    can't disagree about how a model type is made.

    Args:
        dctSettings: Must have "type" (dummy, logistic, random_forest or rule) and
            that type's settings, as written in config.yaml.
        intSeed: The random seed (project.random_state).

    Returns:
        An unfitted scikit-learn model.

    Raises:
        ValueError: If the type is not one of the four above.
    """
    strType = dctSettings["type"]

    if strType == "dummy":
        # A baseline that ignores the inputs. "most_frequent" always predicts
        # the commoner target value. A real model has to beat this to be useful.
        return DummyClassifier(strategy=dctSettings["strategy"])

    if strType == "logistic":
        return LogisticRegression(
            max_iter=dctSettings["max_iter"],
            C=dctSettings["C"],
            class_weight=dctSettings["class_weight"],
            random_state=intSeed,
        )

    if strType == "random_forest":
        # n_jobs=1 keeps results identical from run to run and machine to machine.
        return RandomForestClassifier(
            n_estimators=dctSettings["n_estimators"],
            max_depth=dctSettings["max_depth"],
            min_samples_leaf=dctSettings["min_samples_leaf"],
            class_weight=dctSettings["class_weight"],
            random_state=intSeed,
            n_jobs=1,
        )

    if strType == "rule":
        # A fixed rule, such as age at or above 65. Used as a benchmark, not a model.
        return RuleBenchmark(column=dctSettings["column"], threshold=dctSettings["threshold"])

    raise ValueError(f"Unknown model type '{strType}'. Use dummy, logistic, random_forest or rule.")


def mdlTrainModel(dfRawTrain, serYTrain, dctConfig: dict):
    """Fit the feature step and the model together on the training rows, and return the pipeline.

    The returned pipeline has two steps, "features" and "model". Given raw patient
    columns, it builds the calculated and encoded columns and then scores them, so
    one saved file does both jobs.

    Keeping this as its own small function means tests can train a model on
    tiny data without touching the disk.

    Args:
        dfRawTrain: Raw training inputs (the feature_columns, before any engineering).
        serYTrain: Training targets.
        dctConfig: The loaded config.yaml settings. Uses model.max_iter, model.C,
            model.class_weight and project.random_state.

    Returns:
        The fitted pipeline.
    """
    dctModelSettings = dctConfig["model"]

    # The saved model is a logistic regression built from the model section.
    # max_iter caps the solver's steps. C sets the regularisation strength.
    # class_weight "balanced" gives the rarer target value more weight (None means
    # equal weight). random_state keeps results repeatable.
    mdlLogistic = mdlBuildModel({"type": "logistic", **dctModelSettings}, dctConfig["project"]["random_state"])
    mdlPipeline = Pipeline([("features", objBuildFeatureStep(dctConfig)), ("model", mdlLogistic)])
    mdlPipeline.fit(dfRawTrain, serYTrain)
    return mdlPipeline


def arrCrossValScores(mdlModel, dfXTrain, serYTrain, intFolds: int, intSeed: int):
    """Return the AUC of each cross-validation slice of the training rows.

    The training rows are cut into slices. The model is fitted on all but one
    slice and scored on the one left out, once for each slice. The test rows
    are never touched, so they stay an honest final check.

    Args:
        mdlModel: An unfitted model. It is copied for each slice, so it is
            still unfitted afterwards.
        dfXTrain: Training inputs.
        serYTrain: Training targets.
        intFolds: How many slices (comparison.cv_folds).
        intSeed: Random seed for cutting the slices (project.random_state).

    Returns:
        An array with one AUC per slice.
    """
    # Stratified slices keep the same share of each target value in every slice.
    objFolds = StratifiedKFold(n_splits=intFolds, shuffle=True, random_state=intSeed)
    return cross_val_score(mdlModel, dfXTrain, serYTrain, cv=objFolds, scoring="roc_auc")


def tplCrossValAuc(mdlModel, dfXTrain, serYTrain, intFolds: int, intSeed: int) -> tuple:
    """Score a model by cross-validation on the training rows only.

    Args:
        mdlModel: An unfitted model.
        dfXTrain: Training inputs.
        serYTrain: Training targets.
        intFolds: How many slices (comparison.cv_folds).
        intSeed: Random seed for cutting the slices (project.random_state).

    Returns:
        A tuple of the mean AUC and the standard deviation across the slices.
    """
    arrScores = arrCrossValScores(mdlModel, dfXTrain, serYTrain, intFolds, intSeed)
    return float(arrScores.mean()), float(arrScores.std())


def dctCheckModelGate(dfXTrain, serYTrain, dctConfig: dict) -> dict:
    """Decide whether the model to be saved is good enough to keep.

    The model has to beat the do-nothing baseline on cross-validated AUC by at
    least approval.min_auc_gain. Only training rows are used. The threshold is
    set in the config because it should come from the intended use and the
    cost of errors, not from the code.

    Args:
        dfXTrain: Training inputs.
        serYTrain: Training targets.
        dctConfig: The loaded config.yaml settings. Uses approval, model,
            comparison.baseline, comparison.cv_folds and project.random_state.

    Returns:
        A dictionary with enabled, baseline_cv_auc, model_cv_auc, auc_gain,
        required_gain and passed. When the gate is switched off, passed is True
        and the scores are None.
    """
    dctApproval = dctConfig["approval"]
    if not dctApproval["enabled"]:
        return {"enabled": False, "baseline_cv_auc": None, "model_cv_auc": None,
                "auc_gain": None, "required_gain": dctApproval["min_auc_gain"], "passed": True}

    intSeed = dctConfig["project"]["random_state"]
    intFolds = dctConfig["comparison"]["cv_folds"]
    mdlBaseline = mdlBuildModel(dctConfig["comparison"]["baseline"], intSeed)
    mdlCandidate = mdlBuildModel({"type": "logistic", **dctConfig["model"]}, intSeed)

    fltBaseline, _ = tplCrossValAuc(mdlBaseline, dfXTrain, serYTrain, intFolds, intSeed)
    fltCandidate, _ = tplCrossValAuc(mdlCandidate, dfXTrain, serYTrain, intFolds, intSeed)
    fltGain = fltCandidate - fltBaseline
    return {
        "enabled": True,
        "baseline_cv_auc": fltBaseline,
        "model_cv_auc": fltCandidate,
        "auc_gain": fltGain,
        "required_gain": dctApproval["min_auc_gain"],
        "passed": bool(fltGain >= dctApproval["min_auc_gain"]),
    }


def dctBuildModelCard(dctConfig: dict, dctGate: dict, dfXTrain, serYTrain, strDataPath, strModelPath) -> dict:
    """Collect the facts that identify and describe the saved model.

    The card travels beside the model file so a reviewer can see what the model
    is for, what it was built from and whether it passed the gate, without
    opening any code.

    Args:
        dctConfig: The loaded config.yaml settings. Uses model, features,
            approval and project.
        dctGate: The result from dctCheckModelGate.
        dfXTrain: Engineered training inputs, for the model column names and row count.
        serYTrain: Training targets, for the readmission rate.
        strDataPath: The processed CSV the model was trained from.
        strModelPath: The saved model file, for its fingerprint.

    Returns:
        A dictionary that can be written as JSON.
    """
    dctApproval = dctConfig["approval"]
    return {
        "model_type": "logistic_regression",
        "created_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {strKey: varValue for strKey, varValue in dctConfig["model"].items() if strKey != "path"},
        "target": dctConfig["features"]["target_column"],
        "input_columns": list(dctConfig["features"]["feature_columns"]),
        "model_columns": list(dfXTrain.columns),
        "training_rows": int(len(dfXTrain)),
        "training_positive_rate": round(float(serYTrain.mean()), 4),
        "random_state": dctConfig["project"]["random_state"],
        "model_fingerprint": strFileFingerprint(strModelPath),
        "data_fingerprint": strFileFingerprint(strDataPath),
        "config_fingerprint": strFileFingerprint("config.yaml"),
        "git_commit": strGitCommit(),
        "gate": dctGate,
        "status": ("Passed the automated gate. A named person must still approve it before any real use."
                   if dctGate["passed"] else "Failed the automated gate."),
        "intended_use": dctApproval["intended_use"],
        "not_for": dctApproval["not_for"],
        "known_limits": dctApproval["known_limits"],
    }


def pthSaveModelCard(dctCard: dict, varPath) -> Path:
    """Write the model card as a JSON file and return the path written.

    Args:
        dctCard: The card from dctBuildModelCard.
        varPath: Where to write it, as text or a Path.

    Returns:
        The path that was written.
    """
    pthFile = Path(varPath)
    pthFile.parent.mkdir(parents=True, exist_ok=True)
    pthFile.write_text(json.dumps(dctCard, indent=2), encoding="utf-8")
    return pthFile


def pthSaveFeatures(dfXTrain, dfXTest, serYTrain, serYTest, varPath) -> Path:
    """Save the encoded features, with the target, so you can see what the model saw.

    The file is for inspection only. No stage reads it back, because the stages
    all build features with the same function, and reading a saved copy could
    let training and prediction drift apart.

    Args:
        dfXTrain: Encoded training inputs.
        dfXTest: Encoded test inputs.
        serYTrain: Training targets.
        serYTest: Test targets.
        varPath: Where to write the file, as text or a Path.

    Returns:
        The path that was written. The file has every encoded input column, the
        target column, and a "split" column that says "train" or "test".
    """
    # Build one table for each group, adding the target and the group name.
    dfTrain = dfXTrain.assign(**{serYTrain.name: serYTrain, "split": "train"})
    dfTest = dfXTest.assign(**{serYTest.name: serYTest, "split": "test"})

    pthFile = Path(varPath)
    pthFile.parent.mkdir(parents=True, exist_ok=True)

    # Stack the two groups and write them, without pandas' own row numbers.
    pd.concat([dfTrain, dfTest]).to_csv(pthFile, index=False)
    return pthFile


def pthSaveModel(mdlModel, varPath) -> Path:
    """Save a fitted model to disk and return the path written."""
    pthFile = Path(varPath)

    # Create artifacts/ if it doesn't exist yet.
    pthFile.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(mdlModel, pthFile)
    return pthFile


def mdlLoadModel(varPath):
    """Load a saved model from disk.

    Raises:
        FileNotFoundError: If there is no model at that path. The message tells
            the user the fix, which is to run the training step first.
    """
    pthFile = Path(varPath)
    if not pthFile.exists():
        raise FileNotFoundError(f"No model at {pthFile}. Run src.train first.")
    return joblib.load(pthFile)

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

def tplTrainAndSaveModel(dctConfig: dict, dfRawTrain: pd.DataFrame, serYTrain: pd.Series,
                         strInputPath: str, strModelPath: str) -> tuple:
    """Fit the model, check it against the approval gate, save it and record the run.

    It prints the gate result and where the model and card were saved, so the text
    goes to the stage's RTF file with everything else.

    Args:
        dctConfig: The loaded config.yaml settings.
        dfRawTrain: The raw training inputs (feature_columns only, before engineering).
        serYTrain: The training targets from lstSplitData.
        strInputPath: The processed CSV the model is trained from.
        strModelPath: Where to save the model.

    Returns:
        A tuple of the model file path and the model card path.

    Raises:
        ModelGateError: If the model fails the approval gate. It is not saved.
    """

    print(f"\n{STR_BOLD}{STR_UNDERLINE}Model Training{STR_RESET}")

    # Fit the model and save it for the later stages.
    mdlModel = mdlTrainModel(dfRawTrain, serYTrain, dctConfig)

    # The gate and the card work on the engineered columns, built the same way the pipeline builds them.
    dfXTrain = dfEngineerFeatures(dfRawTrain, dctConfig)

    # The approval gate. The model is saved only if it beats the baseline by
    # the margin set in approval.min_auc_gain. Training rows only.
    dctGate = dctCheckModelGate(dfXTrain, serYTrain, dctConfig)
    if dctGate["enabled"]:
        print(f"\nApproval gate: baseline cross-validated AUC {dctGate['baseline_cv_auc']:.3f}, "
              f"model {dctGate['model_cv_auc']:.3f}, gain {dctGate['auc_gain']:.3f}, "
              f"required {dctGate['required_gain']:.3f}. {'PASSED' if dctGate['passed'] else 'FAILED'}.")
    else:
        print("\nApproval gate is switched off in config.yaml.")
    if not dctGate["passed"]:
        # Stop here. The model is not saved, so evaluate.py and predict.py
        # can't pick up a model that failed.
        raise ModelGateError(
            f"\nThe model gained {dctGate['auc_gain']:.3f} AUC over the baseline, "
            f"below the required {dctGate['required_gain']:.3f}. The model was not saved."
        )

    pthOut = pthSaveModel(mdlModel, strModelPath)
    print(f"\nTrained on {len(dfXTrain)} rows. Model saved to {pthOut}")

    # Save the model card beside the model: what it is for, what built it, and the gate result.
    pthCard = pthSaveModelCard(
        dctBuildModelCard(dctConfig, dctGate, dfXTrain, serYTrain, strInputPath, pthOut),
        dctConfig["approval"]["model_card_path"],
    )
    print(f"\nModel card saved to {pthCard}\n")

    # Record this training run. The model's fingerprint lets evaluate.py and
    # compare.py point back to exactly this model file.
    dctFeatures = dctConfig["features"]
    bLogRun(
        dctConfig, "train", "logistic_regression",
        dctParams={
            **{f"model.{strKey}": varValue for strKey, varValue in dctConfig["model"].items() if strKey != "path"},
            "features.feature_columns": dctFeatures["feature_columns"],
            "features.categorical_columns": dctFeatures["categorical_columns"],
            "features.difference_columns": dctFeatures["difference_columns"],
            "features.test_size": dctFeatures["test_size"],
            "features.stratify": dctFeatures["stratify"],
            "project.random_state": dctConfig["project"]["random_state"],
        },
        dctMetrics={"training_rows": len(dfXTrain)},
        lstArtifacts=[pthOut, pthCard],
        dctTags={"model_fingerprint": strFileFingerprint(pthOut), "gate_passed": str(dctGate["passed"])},
    )
    return pthOut, pthCard


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


def subMain() -> None:
    """Command line entry point: run the full training stage."""
    objParser = argparse.ArgumentParser(description="Train a model and save it.")
    subAddConfigArgument(objParser)
    objParser.add_argument("--input", default=None, help="CSV to train on (default: data.processed_path)")
    objParser.add_argument("--model", default=None, help="Where to save the model (default: model.path)")
    objArgs = objParser.parse_args()

    dctConfig = dctLoadConfig(objArgs.config)
    strInputPath = objArgs.input or dctConfig["data"]["processed_path"]
    strModelPath = objArgs.model or dctConfig["model"]["path"]

    # Everything printed inside this block goes to the RTF file, not the screen.
    with objCaptureToRtf(dctConfig["outputs"]["rtf_path"], "Train"):
        # Display the file header at the top of the stage output
        subPrintPiplineHeader(
            "Train",
            dctConfig["project"]["client_question"],
            dctConfig["features"]["target_column"],
            ", ".join(dctConfig["features"]["feature_columns"]),
            strInputPath,
        )

        # Load the data and stop immediately if it fails validation.
        dfData = dfValidate(dfLoadCsv(strInputPath), dctConfig)

        # Build dfX and serY, then split. The test rows are thrown away here (the
        # underscores) because evaluate.py is the stage that uses them.
        dfX, serY = tplBuildFeatures(dfData, dctConfig)
        dfXTrain, dfXTest, serYTrain, serYTest = lstSplitData(dfX, serY, dctConfig)

        # Save the encoded features so you can open the file and see what the model saw.
        # The test rows are only written to the file for inspection. They are not used here.
        pthFeatures = pthSaveFeatures(dfXTrain, dfXTest, serYTrain, serYTest, dctConfig["outputs"]["features_path"])
        print(f"\nEncoded features ({dfXTrain.shape[1]} inputs) saved to {pthFeatures}")

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

        pthOut, pthCard = tplTrainAndSaveModel(
            dctConfig, dfRawInputs(dfData, dctConfig).loc[dfXTrain.index], serYTrain, strInputPath, strModelPath)

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
            "",
        )

        # The saved model above is only a file. Say who must approve it, and what checks, before anyone uses it.
        subPrintStep(
            "Step 6 - Identify appropriate human oversight, governance and approval requirements before real-world use.",
            "",
        )


if __name__ == "__main__":
    subMain()
