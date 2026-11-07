"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    compare.py

    Stage 6 of the pipeline: model comparison (benchmarking).

    Trains a baseline, one or more simple rule benchmarks and one or more candidate
    models on exactly the same training rows, scores them all on exactly the same
    test rows, and writes one table of results. A candidate is only worth keeping
    if it beats the baseline and the rules. It also writes an evaluation evidence
    record (Markdown): the intended use, the table, which model is recommended and
    why, and the known limits.

    The baseline is a model that ignores the inputs and always predicts the commoner
    outcome. With about 14 percent of patients readmitted, it is right about 86
    percent of the time while never finding a single readmission. That is why
    accuracy alone can't be trusted here, and why the table also shows precision,
    recall and AUC.

    Each model is also scored with cross-validation on the training rows only. That
    repeats the fit and the score several times on different slices of the training
    data, so you can see how steady the AUC is. The test rows are used once, at the
    end, so they stay an honest check.

    The models and their settings come from config.yaml (the comparison section).
    The split is the same one train.py and evaluate.py use.

    The printed output is saved to an RTF file instead of the screen. The file name
    comes from outputs.rtf_path in config.yaml (the text {stage} becomes the stage name).

    Run from the project root:
        python -m src.compare
        python -m src.compare --output reports/other_table.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from src.config import subAddConfigArgument, dctLoadConfig
# Saves what the stage prints to an RTF file instead of the screen.
from src.display import (objCaptureToRtf, bDisplayDatasetDetails, subPrintPiplineHeader,
                         STR_BOLD, STR_UNDERLINE, STR_RESET, STR_ITALIC)
# Records each model as a run in MLflow when tracking is on.
from src.tracking import bLogRun
from src.evaluate import dctComputeMetrics
from src.features import tplBuildFeatures, lstSplitData
from src.ingest import dfLoadCsv
from src.train import mdlBuildModel, tplCrossValAuc
from src.validate import dfValidate


def dfCompareModels(dfXTrain, dfXTest, serYTrain, serYTest, dctConfig: dict) -> pd.DataFrame:
    """Train the baseline and every candidate, and return one row of scores for each.

    Args:
        dfXTrain: Training inputs.
        dfXTest: Test inputs. Only used after each model is fitted.
        serYTrain: Training targets.
        serYTest: Test targets.
        dctConfig: The loaded config.yaml settings. Uses comparison.baseline,
            comparison.benchmarks, comparison.candidates, comparison.cv_folds, project.random_state and
            evaluation.positive_label.

    Returns:
        A DataFrame with one row per model: its name, role (baseline, benchmark or candidate),
        type, cross-validated AUC (mean and standard deviation), and the test-set
        accuracy, f1, precision, recall and roc_auc.
    """
    intSeed = dctConfig["project"]["random_state"]
    dctComparison = dctConfig["comparison"]
    varPositive = dctConfig["evaluation"]["positive_label"]

    # The baseline goes first, then any rule benchmarks, then each candidate in the order they appear in the config.
    lstModels = [("baseline", "baseline", dctComparison["baseline"])]
    lstModels += [(strName, "benchmark", dctSettings) for strName, dctSettings in dctComparison["benchmarks"].items()]
    lstModels += [(strName, "candidate", dctSettings) for strName, dctSettings in dctComparison["candidates"].items()]

    lstRows = []
    for strName, strRole, dctSettings in lstModels:
        mdlModel = mdlBuildModel(dctSettings, intSeed)

        # Cross-validation on the training rows only. Every model uses the same
        # slices (same folds and seed), so they are compared fairly.
        fltCvMean, fltCvStd = tplCrossValAuc(mdlModel, dfXTrain, serYTrain, dctComparison["cv_folds"], intSeed)

        # Now fit on all the training rows and score once on the test rows.
        mdlModel.fit(dfXTrain, serYTrain)
        intPositive = list(mdlModel.classes_).index(varPositive)
        arrProbabilities = mdlModel.predict_proba(dfXTest)[:, intPositive]
        dctMetrics = dctComputeMetrics(serYTest, mdlModel.predict(dfXTest), dctConfig, arrProbabilities)

        lstRows.append({
            "model": strName,
            "role": strRole,
            "type": dctSettings["type"],
            "cv_auc_mean": fltCvMean,
            "cv_auc_std": fltCvStd,
            **dctMetrics,
        })

    return pd.DataFrame(lstRows)


def strBuildEvidenceRecord(dfResults: pd.DataFrame, dctConfig: dict) -> str:
    """Write the evaluation evidence record as Markdown text.

    The record pulls the decision together in one place: what the model is for,
    how it scored against the baseline and the rules, which candidate is
    recommended and why, and what the results do not establish.

    The recommended candidate is the simplest one (first in the config) whose
    cross-validated AUC is within one standard deviation of the best. A tiny gap
    between models is noise, so the simpler model wins.

    Args:
        dfResults: The table from dfCompareModels.
        dctConfig: The loaded config.yaml settings. Uses approval.

    Returns:
        The record as one string, ready to write to a file.
    """
    dctApproval = dctConfig["approval"]
    dfCandidates = dfResults[dfResults["role"] == "candidate"].reset_index(drop=True)
    serBest = dfCandidates.loc[dfCandidates["cv_auc_mean"].idxmax()]

    # The simplest candidate that is within one standard deviation of the best.
    dfClose = dfCandidates[dfCandidates["cv_auc_mean"] >= serBest["cv_auc_mean"] - serBest["cv_auc_std"]]
    serChosen = dfClose.iloc[0]

    serBaseline = dfResults[dfResults["role"] == "baseline"].iloc[0]
    fltGain = serChosen["cv_auc_mean"] - serBaseline["cv_auc_mean"]

    lstLines = [
        "# Evaluation evidence record",
        "",
        "## Intended use",
        dctApproval["intended_use"],
        "",
        "## Not for",
        dctApproval["not_for"],
        "",
        "## Results",
        "Cross-validated AUC uses the training rows only. The other scores use the held-out test rows, measured once.",
        "",
        "| Model | Role | CV AUC | Test AUC | Accuracy | Recall | Precision |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, serRow in dfResults.iterrows():
        lstLines.append(
            f"| {serRow['model']} | {serRow['role']} | {serRow['cv_auc_mean']:.3f} | {serRow['roc_auc']:.3f} | "
            f"{serRow['accuracy']:.3f} | {serRow['recall']:.3f} | {serRow['precision']:.3f} |"
        )

    lstLines += ["", "## What the results show"]
    lstLines.append(f"Recommended candidate: {serChosen['model']} (cross-validated AUC {serChosen['cv_auc_mean']:.3f}).")
    if len(dfClose) > 1:
        lstLines.append(
            f"{', '.join(dfClose['model'])} are within one standard deviation of each other, so the "
            "difference between them is noise. The simpler one is recommended."
        )
    lstLines.append(
        f"It beats the do-nothing baseline by {fltGain:.3f} AUC. The approval gate requires {dctApproval['min_auc_gain']:.3f}."
    )
    for _, serRule in dfResults[dfResults["role"] == "benchmark"].iterrows():
        strVerdict = "beats" if serChosen["cv_auc_mean"] > serRule["cv_auc_mean"] else "does not beat"
        lstLines.append(
            f"It {strVerdict} the rule '{serRule['model']}' on AUC "
            f"({serChosen['cv_auc_mean']:.3f} against {serRule['cv_auc_mean']:.3f})."
        )

    lstLines += ["", "## What the results do not establish"]
    lstLines += [f"- {strLimit}" for strLimit in dctApproval["known_limits"]]
    lstLines += [
        "",
        "## Recommendation",
        dctApproval["recommendation"],
        "",
    ]
    return "\n".join(lstLines)


def subMain() -> None:
    """Command line entry point: compare the baseline and candidates, and save the table."""
    objParser = argparse.ArgumentParser(description="Compare a baseline with candidate models.")
    subAddConfigArgument(objParser)
    objParser.add_argument("--input", default=None, help="CSV to use (default: data.processed_path)")
    objParser.add_argument("--output", default=None, help="Results table (default: outputs.comparison_path)")
    objArgs = objParser.parse_args()

    dctConfig = dctLoadConfig(objArgs.config)
    strInputPath = objArgs.input or dctConfig["data"]["processed_path"]
    strOutputPath = objArgs.output or dctConfig["outputs"]["comparison_path"]

    # Everything printed inside this block goes to the RTF file, not the screen.
    with objCaptureToRtf(dctConfig["outputs"]["rtf_path"], "Compare"):
        # print the report header
        subPrintPiplineHeader(
            "Compare",
            dctConfig["project"]["client_question"],
            dctConfig["features"]["target_column"],
            ", ".join(dctConfig["features"]["feature_columns"]),
            strInputPath,
        )

        # Same steps as train.py and evaluate.py, so every stage uses the same rows.
        dfData = dfValidate(dfLoadCsv(strInputPath), dctConfig)
        dfX, serY = tplBuildFeatures(dfData, dctConfig)
        dfXTrain, dfXTest, serYTrain, serYTest = lstSplitData(dfX, serY, dctConfig)

        dfResults = dfCompareModels(dfXTrain, dfXTest, serYTrain, serYTest, dctConfig)

        # Save the table, creating the folder if needed.
        pthOut = Path(strOutputPath)
        pthOut.parent.mkdir(parents=True, exist_ok=True)
        dfResults.round(4).to_csv(pthOut, index=False)

        print(f"\n{STR_BOLD}{STR_UNDERLINE}Training Information{STR_RESET}")
        print(f"\n{STR_BOLD}Training rows:{STR_RESET} {len(dfXTrain)}.")
        print(f"\n{STR_BOLD}Test rows:{STR_RESET} {len(dfXTest)}.")
        print(f"\n{STR_BOLD}Cross-validation folds:{STR_RESET} {dctConfig['comparison']['cv_folds']}.")
        print(f"\n{STR_BOLD}Inputs the models see ({dfXTrain.shape[1]}):{STR_RESET} {', '.join(dfXTrain.columns)}\n")
        print(dfResults.round(3).to_string(index=False))
        print(f"\n{STR_BOLD}cv_auc_mean and cv_auc_std:{STR_RESET} AUC across the cross-validation slices of the training rows.")
        print(f"\n{STR_BOLD}accuracy to roc_auc:{STR_RESET} scores on the held-out test rows, measured once.")

        # Name the best candidate by cross-validated AUC (not by the test rows, so the test stays unseen).
        dfCandidates = dfResults[dfResults["role"] == "candidate"]
        serBest = dfCandidates.loc[dfCandidates["cv_auc_mean"].idxmax()]
        print(f"\n{STR_BOLD}Highest cross-validated AUC among the candidates:{STR_RESET} {serBest['model']} ({serBest['cv_auc_mean']:.3f}).")
        print(f"\n{STR_BOLD}Results table saved to: {STR_RESET} {pthOut}")

        # Write the evaluation evidence record beside the other reports.
        pthEvidence = Path(dctConfig["approval"]["evidence_path"])
        pthEvidence.parent.mkdir(parents=True, exist_ok=True)
        pthEvidence.write_text(strBuildEvidenceRecord(dfResults, dctConfig), encoding="utf-8")
        print(f"\n{STR_BOLD}Evaluation evidence record saved to:{STR_RESET} {pthEvidence}")

        # Record one run per model, so the baseline and candidates can be compared in MLflow.
        dctSettings = {
            "baseline": dctConfig["comparison"]["baseline"],
            **dctConfig["comparison"]["benchmarks"],
            **dctConfig["comparison"]["candidates"],
        }
        for _, serRow in dfResults.iterrows():
            dctMetricsRow = {strKey: serRow[strKey] for strKey in serRow.index if strKey not in ("model", "role", "type")}
            bLogRun(
                dctConfig, "compare", serRow["model"],
                dctParams={f"model.{strKey}": varValue for strKey, varValue in dctSettings[serRow["model"]].items()},
                dctMetrics=dctMetricsRow,
                lstArtifacts=[pthOut, pthEvidence],
                dctTags={"role": serRow["role"]},
            )


if __name__ == "__main__":
    subMain()
