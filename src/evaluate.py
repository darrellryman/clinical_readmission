"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    evaluate.py
    
    Stage 5 of the pipeline: model evaluation.
    
    Loads the saved model, scores it on the held-out test rows (the ones train.py
    never saw), and writes the results to a JSON file.
    
    Because the split uses the same random seed from config.yaml, the test rows here
    are exactly the ones set aside during training. That keeps the scores honest.
    
    The model location and metrics file come from config.yaml (model.path and
    outputs.metrics_path).
    
    The printed output is saved to an RTF file instead of the screen. The file name
    comes from outputs.rtf_path in config.yaml (the text {stage} becomes the stage name).

    Run from the project root:
        python -m src.evaluate
"""

import argparse

# json is used to write the metrics in a plain, readable file format.
import json


from pathlib import Path

# pandas builds the small result tables (error counts and scores by group).
import pandas as pd

# Standard scikit-learn scoring functions.
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)

from src.config import subAddConfigArgument, dctLoadConfig
# Saves what the stage prints to an RTF file instead of the screen.
from src.display import (objCaptureToRtf, bDisplayDatasetDetails, subPrintPiplineHeader,
                         STR_BOLD, STR_UNDERLINE, STR_RESET, STR_ITALIC)
# Records the run (scores, files, tags) in MLflow when tracking is on.
from src.tracking import bLogRun, strFileFingerprint
from src.features import tplBuildFeatures, lstSplitData, dfRawInputs
from src.ingest import dfLoadCsv
from src.train import mdlLoadModel
from src.validate import dfValidate


def dctComputeMetrics(varYTrue, varYPred, dctConfig: dict, varProbabilities=None) -> dict[str, float]:
    """Compare predictions with the real answers and return the scores.

    Args:
        varYTrue: The correct answers (a list, array or Series).
        varYPred: The model's predictions (a list, array or Series).
        dctConfig: The loaded config.yaml settings. Uses evaluation.average and
            evaluation.positive_label.
        varProbabilities: Optional. The model's probability that each row is the
            positive class. When given, roc_auc is added to the scores.

    Returns:
        A dictionary with accuracy, f1, precision and recall, and roc_auc when
        probabilities are given.
        - Accuracy is the share of predictions that were right. When one outcome is
          rare, accuracy can look good even if the model never predicts it.
        - F1 balances precision and recall. How it is averaged across classes is
          the evaluation.average setting. The default, "weighted", weights each
          class by how common it is, so it still means something when classes are uneven.
        - Precision is the share of patients flagged as positive who really were.
        - Recall is the share of real positive patients the model caught.
          roc_auc is how well the model ranks positives above negatives. 0.5 ranks the
          readmitted patient higher only half the time and 1.0 is perfect.
    """
    strAverage = dctConfig["evaluation"]["average"]
    varPositive = dctConfig["evaluation"]["positive_label"]

    # float() converts numpy numbers into plain Python floats, which json can write.
    # zero_division=0 gives a score of 0 (and no warning) when the model never
    # predicts the positive value.
    dctMetrics = {
        "accuracy": float(accuracy_score(varYTrue, varYPred)),
        "f1": float(f1_score(varYTrue, varYPred, average=strAverage)),
        "precision": float(precision_score(varYTrue, varYPred, pos_label=varPositive, zero_division=0)),
        "recall": float(recall_score(varYTrue, varYPred, pos_label=varPositive, zero_division=0)),
    }
    if varProbabilities is not None:
        dctMetrics["roc_auc"] = float(roc_auc_score(varYTrue, varProbabilities))
    return dctMetrics


def dfConfusionCounts(varYTrue, varYPred, dctConfig: dict) -> pd.DataFrame:
    """Count each kind of right and wrong prediction.

    Args:
        varYTrue: The correct answers.
        varYPred: The model's predictions.
        dctConfig: The loaded config.yaml settings. Uses evaluation.positive_label.

    Returns:
        A one-row table with four counts. A true positive is a readmitted patient
        the model flagged. A false negative is a readmitted patient it missed. A
        false positive is a patient it flagged who was not readmitted. A true
        negative is a patient it correctly left alone.
    """
    varPositive = dctConfig["evaluation"]["positive_label"]

    # Put the negative value first and the positive value second, so the
    # four numbers always come out in the same places.
    lstLabels = sorted(set(varYTrue) | set(varYPred), key=lambda varLabel: varLabel == varPositive)
    arrCounts = confusion_matrix(varYTrue, varYPred, labels=lstLabels)

    # Raises if the target is not a two-value yes or no, because the four counts
    # only make sense for that.
    if arrCounts.shape != (2, 2):
        raise ValueError("Error counts need a target with exactly two values")

    intTn, intFp, intFn, intTp = (int(varCount) for varCount in arrCounts.ravel())
    return pd.DataFrame([{
        "true_positive": intTp, "false_negative": intFn,
        "false_positive": intFp, "true_negative": intTn,
    }])


def dfBuildGroups(dfData: pd.DataFrame, dfXTest: pd.DataFrame, dctConfig: dict) -> pd.DataFrame:
    """Pick out the group labels for each test row.

    Args:
        dfData: The full cleaned data, with the original columns (text sex, age).
        dfXTest: The test inputs. Only its row labels are used, to find the
            matching rows in dfData.
        dctConfig: The loaded config.yaml settings. Uses evaluation.group_columns
            and evaluation.binned_columns.

    Returns:
        A DataFrame with one column per grouping and one row per test row. Binned
        columns hold band labels such as "40-60".

    Raises:
        KeyError: If a grouping column is not in dfData.
    """
    dctEvaluation = dctConfig["evaluation"]
    dfGroups = pd.DataFrame(index=dfXTest.index)

    # Plain columns: each distinct value is its own group.
    for strColumn in dctEvaluation["group_columns"] or []:
        dfGroups[strColumn] = dfData.loc[dfXTest.index, strColumn].astype(str)

    # Numeric columns: cut into bands first. include_lowest keeps a value that
    # sits exactly on the first edge (such as age 0).
    for strColumn, lstEdges in (dctEvaluation["binned_columns"] or {}).items():
        # The labels come from the config edges, so the first band reads "0-40" and not "-0.001-40".
        lstLabels = [f"{lstEdges[intPos]:g}-{lstEdges[intPos + 1]:g}" for intPos in range(len(lstEdges) - 1)]
        serBands = pd.cut(dfData.loc[dfXTest.index, strColumn], bins=lstEdges, labels=lstLabels, include_lowest=True)
        dfGroups[strColumn] = serBands.astype(str)

    return dfGroups


def dfMetricsByGroup(dfGroups: pd.DataFrame, varYTrue, varYPred, varProbabilities, dctConfig: dict) -> pd.DataFrame:
    """Score the model separately for each group, to see whether errors are uneven.

    Args:
        dfGroups: Group labels for each test row, from dfBuildGroups().
        varYTrue: The correct answers, in the same row order as dfGroups.
        varYPred: The model's predictions, in the same order.
        varProbabilities: The model's probability of the positive class.
        dctConfig: The loaded config.yaml settings. Uses evaluation.positive_label.

    Returns:
        One row per group with its size, the real readmission rate, the share the
        model flagged, precision, recall and roc_auc. roc_auc is blank for a
        group that has only one kind of outcome, because it can't be worked out.
    """
    varPositive = dctConfig["evaluation"]["positive_label"]
    serTrue = pd.Series(list(varYTrue), index=dfGroups.index)
    serPred = pd.Series(list(varYPred), index=dfGroups.index)
    serProb = pd.Series(list(varProbabilities), index=dfGroups.index)

    lstRows = []
    for strColumn in dfGroups.columns:
        for strLevel in sorted(dfGroups[strColumn].unique()):
            serMask = dfGroups[strColumn] == strLevel
            serGroupTrue = serTrue[serMask]
            lstRows.append({
                "group": f"{strColumn}={strLevel}",
                "rows": int(serMask.sum()),
                "actual_rate": float((serGroupTrue == varPositive).mean()),
                "flagged_rate": float((serPred[serMask] == varPositive).mean()),
                "precision": float(precision_score(serGroupTrue, serPred[serMask], pos_label=varPositive, zero_division=0)),
                "recall": float(recall_score(serGroupTrue, serPred[serMask], pos_label=varPositive, zero_division=0)),
                # AUC needs both outcomes present in the group.
                "roc_auc": float(roc_auc_score(serGroupTrue, serProb[serMask])) if serGroupTrue.nunique() > 1 else float("nan"),
            })
    return pd.DataFrame(lstRows)


def subMain() -> None:
    """Command line entry point: score the saved model and write the metrics."""
    objParser = argparse.ArgumentParser(description="Evaluate the saved model.")
    subAddConfigArgument(objParser)
    objParser.add_argument("--input", default=None, help="CSV to evaluate on (default: data.processed_path)")
    objParser.add_argument("--model", default=None, help="Saved model (default: model.path)")
    objParser.add_argument("--output", default=None, help="Metrics file (default: outputs.metrics_path)")
    objArgs = objParser.parse_args()

    dctConfig = dctLoadConfig(objArgs.config)
    strInputPath = objArgs.input or dctConfig["data"]["processed_path"]
    strModelPath = objArgs.model or dctConfig["model"]["path"]
    strOutputPath = objArgs.output or dctConfig["outputs"]["metrics_path"]

    # Everything printed inside this block goes to the RTF file, not the screen.
    with objCaptureToRtf(dctConfig["outputs"]["rtf_path"], "Evaluate"):
        # print the report header
        subPrintPiplineHeader(
            "Evaluate",
            dctConfig["project"]["client_question"],
            dctConfig["features"]["target_column"],
            ", ".join(dctConfig["features"]["feature_columns"]),
            strInputPath,
        )

        # Rebuild the same split that train.py used. This time we keep the test
        # rows and ignore the training rows.
        dfData = dfValidate(dfLoadCsv(strInputPath), dctConfig)
        dfX, serY = tplBuildFeatures(dfData, dctConfig)
        _, dfXTest, serYTrain, serYTest = lstSplitData(dfX, serY, dctConfig)

        # Load the trained model and score it on the unseen test rows.
        mdlModel = mdlLoadModel(strModelPath)
        # Find which probability column belongs to the positive value (for AUC).
        intPositive = list(mdlModel.classes_).index(dctConfig["evaluation"]["positive_label"])
        # The saved pipeline builds the engineered columns itself, so it is given the raw test rows.
        dfRawTest = dfRawInputs(dfData, dctConfig).loc[dfXTest.index]
        arrProbabilities = mdlModel.predict_proba(dfRawTest)[:, intPositive]
        dctMetrics = dctComputeMetrics(serYTest, mdlModel.predict(dfRawTest), dctConfig, arrProbabilities)

        # Write the metrics to disk, creating the folder if needed.
        pthOut = Path(strOutputPath)
        pthOut.parent.mkdir(parents=True, exist_ok=True)
        pthOut.write_text(json.dumps(dctMetrics, indent=2))
        print(f"\n{STR_BOLD}{STR_UNDERLINE}Metrics Information{STR_RESET}")
        print(f"\n{STR_BOLD}Metrics written to:{STR_RESET} {pthOut}")
        print(f"\n{STR_BOLD}Metrics results:{STR_RESET} {dctMetrics}")

        # Record the scores. The model fingerprint ties them to the model file they came from.
        bLogRun(
            dctConfig, "evaluate", "logistic_regression_test_scores",
            dctParams={"evaluation.average": dctConfig["evaluation"]["average"],
                       "evaluation.positive_label": dctConfig["evaluation"]["positive_label"],
                       "test_rows": len(dfXTest)},
            dctMetrics=dctMetrics,
            lstArtifacts=[pthOut],
            dctTags={"model_fingerprint": strFileFingerprint(strModelPath)},
        )

        # Count each kind of error, so the cost of a miss or a false alarm can be discussed.
        arrPredictions = mdlModel.predict(dfRawTest)
        print(f"\n{STR_BOLD}{STR_UNDERLINE}Error Counts On The Test Rows{STR_RESET}\n")
        print(dfConfusionCounts(serYTest, arrPredictions, dctConfig).to_string(index=False))

        # Score each group separately. Needs the original columns, so it uses dfData.
        dfGroups = dfBuildGroups(dfData, dfXTest, dctConfig)
        if len(dfGroups.columns) > 0:
            print(f"\n{STR_BOLD}{STR_UNDERLINE}Scores By Group{STR_RESET}\n")
            print(dfMetricsByGroup(dfGroups, serYTest, arrPredictions, arrProbabilities, dctConfig).round(3).to_string(index=False))

        dctCard = json.loads(Path(dctConfig["approval"]["model_card_path"]).read_text(encoding="utf-8"))
        print(f"\n{STR_BOLD}Model card status:{STR_RESET} {dctCard['status']}")
        print(f"\n{STR_BOLD}Intended use:{STR_RESET} {dctCard['intended_use']}")
        print(f"\n{STR_BOLD}Not for:{STR_RESET} {dctCard['not_for']}")
        print(
            f"\n{STR_BOLD}Recall / precision on test rows:{STR_RESET} {dctMetrics['recall']:.3f} / {dctMetrics['precision']:.3f}")


if __name__ == "__main__":
    subMain()
