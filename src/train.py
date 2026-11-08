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

# joblib saves and loads Python objects (like a fitted model) to and from disk.
# It is the standard choice for scikit-learn models.
import joblib

# pandas joins the training and test tables into one file.
import numpy as np
import pandas as pd

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
from src.display import (objCaptureToRtf, bDisplayDatasetDetails, subPrintPiplineHeader,
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
        "status": ("Passed the automated gate. Awaiting sign-off: run python -m src.approve before any real use."
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

        pthOut, pthCard = tplTrainAndSaveModel(
            dctConfig, dfRawInputs(dfData, dctConfig).loc[dfXTrain.index], serYTrain, strInputPath, strModelPath)


if __name__ == "__main__":
    subMain()
