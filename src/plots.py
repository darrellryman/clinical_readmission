"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    plots.py

    Optional stage: draw the evidence graphs.

    Four PNG files are written, each one answering a question a reader would ask:
        1_roc_pr.png          Does the model find readmissions better than chance?
        2_cv_spread.png       Is the gap between models bigger than the noise?
        3_confusion.png       What does a wrong prediction cost in patients?
        4_groups.png          Are the errors spread evenly across groups?

    Nothing is calculated differently here. The split, the models, the folds and
    the group bands are the ones train.py, evaluate.py and compare.py use, so every
    number on a graph matches a number in the reports. The confusion matrix and the
    group graph use the saved model (the one that would be served). The curves and
    the fold spread use the models in the comparison section of config.yaml.

    The folder, picture sharpness and colours come from config.yaml (plots section).
    matplotlib is needed: pip install -r requirements.txt

    The printed output is saved to an RTF file instead of the screen. The file name
    comes from outputs.rtf_path in config.yaml (the text {stage} becomes the stage name).

    Run from the project root, after python -m src.train:
        python -m src.plots
"""

import argparse
from pathlib import Path

import pandas as pd

# The Agg backend draws straight to files, so no window is needed (works on a server or in CI).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from src.config import subAddConfigArgument, dctLoadConfig
from src.display import (objCaptureToRtf, bDisplayDatasetDetails, subPrintStep,
                         STR_BOLD, STR_UNDERLINE, STR_RESET, STR_ITALIC)
from src.evaluate import dfBuildGroups, dfConfusionCounts, dfMetricsByGroup
from src.features import tplBuildFeatures, lstSplitData, dfRawInputs
from src.ingest import dfLoadCsv
from src.train import arrCrossValScores, mdlBuildModel, mdlLoadModel
from src.validate import dfValidate

# File names, in the order they are drawn. The number keeps them sorted in a folder.
STR_ROC_FILE = "1_roc_pr.png"
STR_CV_FILE = "2_cv_spread.png"
STR_CONFUSION_FILE = "3_confusion.png"
STR_GROUPS_FILE = "4_groups.png"


def lstModelSettings(dctConfig: dict) -> list:
    """List every model in the comparison, in table order.

    Args:
        dctConfig: The loaded config.yaml settings. Uses comparison.baseline,
            comparison.benchmarks and comparison.candidates.

    Returns:
        A list of (name, settings) pairs: the baseline first, then the benchmarks,
        then the candidates, in the order they appear in the config.
    """
    dctComparison = dctConfig["comparison"]
    lstModels = [("baseline", dctComparison["baseline"])]
    lstModels += list(dctComparison["benchmarks"].items())
    lstModels += list(dctComparison["candidates"].items())
    return lstModels


def strColour(dctConfig: dict, intPosition: int) -> str:
    """Pick the colour for the model at a position in the table.

    Args:
        dctConfig: The loaded config.yaml settings. Uses plots.palette.
        intPosition: The model's position in the comparison table, from 0.

    Returns:
        A colour code. The palette repeats if there are more models than colours.
    """
    lstPalette = dctConfig["plots"]["palette"]
    return lstPalette[intPosition % len(lstPalette)]


def objPlotRocAndPr(dctProbabilities: dict, serYTest, dctConfig: dict):
    """Draw the ROC curve and the precision-recall curve for every model.

    The ROC curve shows how many readmissions are found as the alert threshold
    loosens, against how many false alarms it costs. A model with no skill runs
    along the diagonal. The precision-recall curve matters more for a rare
    outcome: its dashed line is the share of patients readmitted, which is the
    precision you get by flagging everyone. The baseline is left out of this
    panel because it scores every row the same, so it has no curve.

    Args:
        dctProbabilities: Model name -> probability of readmission for each test row.
        serYTest: The true outcomes for the test rows.
        dctConfig: The loaded config.yaml settings. Uses plots.palette,
            evaluation.positive_label and the comparison order.

    Returns:
        A matplotlib Figure with the two panels side by side.
    """
    varPositive = dctConfig["evaluation"]["positive_label"]
    serPositive = (pd.Series(list(serYTest)) == varPositive).astype(int)
    objFig, lstAxes = plt.subplots(1, 2, figsize=(10, 4.2))

    for intPos, (strName, _) in enumerate(lstModelSettings(dctConfig)):
        arrProb = dctProbabilities[strName]
        arrFpr, arrTpr, _ = roc_curve(serPositive, arrProb)
        fltAuc = roc_auc_score(serPositive, arrProb)
        lstAxes[0].plot(arrFpr, arrTpr, color=strColour(dctConfig, intPos), lw=2, label=f"{strName} ({fltAuc:.3f})")
        # The baseline gives every row the same score, so its precision-recall
        # curve is a meaningless straight line. The dashed base-rate line stands in for it.
        if intPos > 0:
            arrPrecision, arrRecall, _ = precision_recall_curve(serPositive, arrProb)
            lstAxes[1].plot(arrRecall, arrPrecision, color=strColour(dctConfig, intPos), lw=2)

    lstAxes[0].plot([0, 1], [0, 1], "--", color="#bbbbbb")
    lstAxes[0].set_xlabel("False positive rate (false alarms)")
    lstAxes[0].set_ylabel("True positive rate (readmissions found)")
    lstAxes[0].set_title("ROC curves on the test rows (AUC)")
    lstAxes[0].legend(frameon=False, fontsize=8, loc="lower right")

    fltBaseRate = float(serPositive.mean())
    lstAxes[1].axhline(fltBaseRate, ls="--", color="#bbbbbb")
    lstAxes[1].set_xlabel("Recall (readmissions found)")
    lstAxes[1].set_ylabel("Precision (flags that were right)")
    lstAxes[1].set_title(f"Precision-recall (dashed line = readmission rate {fltBaseRate:.2f})")
    lstAxes[1].set_ylim(0, 1)

    for objAx in lstAxes:
        objAx.spines[["top", "right"]].set_visible(False)
    objFig.tight_layout()
    return objFig


def objPlotCvSpread(dctFoldScores: dict, dctConfig: dict):
    """Draw the cross-validated AUC of each fold for every model.

    Each dot is one fold. The black bar is the mean plus or minus one standard
    deviation. When two models' bars overlap, the gap between them is within the
    noise of the folds.

    Args:
        dctFoldScores: Model name -> array of AUC, one per fold.
        dctConfig: The loaded config.yaml settings. Uses plots.palette and the
            comparison order.

    Returns:
        A matplotlib Figure.
    """
    objFig, objAx = plt.subplots(figsize=(6.5, 4))
    lstNames = [strName for strName, _ in lstModelSettings(dctConfig)]

    for intPos, strName in enumerate(lstNames):
        arrScores = dctFoldScores[strName]
        objAx.scatter([intPos] * len(arrScores), arrScores, color=strColour(dctConfig, intPos), s=35, zorder=3)
        objAx.errorbar(intPos, arrScores.mean(), yerr=arrScores.std(), color="black", capsize=6, lw=1.5, zorder=4)
        objAx.text(intPos + 0.12, arrScores.mean(), f"{arrScores.mean():.3f}", va="center", fontsize=9)

    objAx.set_xticks(range(len(lstNames)))
    objAx.set_xticklabels(lstNames, fontsize=8)
    objAx.set_ylabel("AUC on each fold")
    objAx.set_title(f"{len(next(iter(dctFoldScores.values())))}-fold cross-validated AUC (bar = mean ± 1 SD)")
    objAx.spines[["top", "right"]].set_visible(False)
    objFig.tight_layout()
    return objFig


def objPlotConfusion(dfCounts: pd.DataFrame):
    """Draw the four error counts as a labelled grid.

    Args:
        dfCounts: The one-row table from dfConfusionCounts() with true_positive,
            false_negative, false_positive and true_negative.

    Returns:
        A matplotlib Figure. Rows are what actually happened, columns what the
        model predicted.
    """
    serCounts = dfCounts.iloc[0]
    arrGrid = [[serCounts["true_negative"], serCounts["false_positive"]],
               [serCounts["false_negative"], serCounts["true_positive"]]]
    fltMax = max(max(lstRow) for lstRow in arrGrid)

    objFig, objAx = plt.subplots(figsize=(4.8, 4.2))
    objAx.imshow(arrGrid, cmap="Blues")
    lstWords = [["correct, left alone", "false alarm"], ["missed readmission", "readmission found"]]
    for intRow in range(2):
        for intCol in range(2):
            strColor = "white" if arrGrid[intRow][intCol] > fltMax / 2 else "black"
            objAx.text(intCol, intRow - 0.08, f"{arrGrid[intRow][intCol]:,}", ha="center", va="center", fontsize=16, color=strColor)
            objAx.text(intCol, intRow + 0.22, lstWords[intRow][intCol], ha="center", va="center", fontsize=8, color=strColor)
    objAx.set_xticks([0, 1])
    objAx.set_xticklabels(["Predicted\nnot readmitted", "Predicted\nreadmitted"])
    objAx.set_yticks([0, 1])
    objAx.set_yticklabels(["Actually\nnot readmitted", "Actually\nreadmitted"])
    objAx.set_title("Error counts on the test rows (saved model)")
    for objSpine in objAx.spines.values():
        objSpine.set_visible(False)
    objFig.tight_layout()
    return objFig


def objPlotGroups(dfByGroup: pd.DataFrame, dctConfig: dict):
    """Draw recall and precision for every group, with the group sizes.

    Args:
        dfByGroup: The table from dfMetricsByGroup(): group, rows, recall, precision...
        dctConfig: The loaded config.yaml settings. Uses plots.group_metrics and
            plots.palette.

    Returns:
        A matplotlib Figure. Each group label shows how many test rows it has, so a
        reader can see that a bar from a small group is less certain.
    """
    lstMetrics = dctConfig["plots"]["group_metrics"]
    intGroups = len(dfByGroup)
    fltWidth = 0.8 / len(lstMetrics)

    objFig, objAx = plt.subplots(figsize=(max(7, 1.1 * intGroups), 4.2))
    for intMetric, strMetric in enumerate(lstMetrics):
        lstX = [intPos + (intMetric - (len(lstMetrics) - 1) / 2) * fltWidth for intPos in range(intGroups)]
        lstBars = objAx.bar(lstX, dfByGroup[strMetric], fltWidth, color=strColour(dctConfig, 2 + intMetric), label=strMetric)
        objAx.bar_label(lstBars, fmt="%.2f", fontsize=7, padding=2)

    objAx.set_xticks(range(intGroups))
    objAx.set_xticklabels([f"{strGroup}\nn={intRows}" for strGroup, intRows in zip(dfByGroup["group"], dfByGroup["rows"])], fontsize=8)
    objAx.set_ylim(0, 1)
    objAx.legend(frameon=False)
    objAx.set_title("Scores by group on the test rows (n = rows in the group)")
    objAx.spines[["top", "right"]].set_visible(False)
    objFig.tight_layout()
    return objFig


def pthSaveFigure(objFig, varFolder, strFileName: str, intDpi: int) -> Path:
    """Write a figure to a PNG file and release its memory.

    Args:
        objFig: The matplotlib Figure to save.
        varFolder: The folder to write to. Created if it does not exist.
        strFileName: The file name, such as "1_roc_pr.png".
        intDpi: Picture sharpness in dots per inch.

    Returns:
        The path of the written file.
    """
    pthFolder = Path(varFolder)
    pthFolder.mkdir(parents=True, exist_ok=True)
    pthOut = pthFolder / strFileName
    objFig.savefig(pthOut, dpi=intDpi)
    # Closing frees the figure. Without it, memory grows with every run.
    plt.close(objFig)
    return pthOut


def lstDrawAllFigures(dfData: pd.DataFrame, mdlSaved, dctConfig: dict) -> list:
    """Rebuild the split, score the models, and write all four graphs.

    Args:
        dfData: The validated clean data, with the original columns.
        mdlSaved: The fitted model loaded from model.path.
        dctConfig: The loaded config.yaml settings.

    Returns:
        A list of the paths written.

    Raises:
        ValueError: If the target does not have exactly two values.
    """
    intSeed = dctConfig["project"]["random_state"]
    varPositive = dctConfig["evaluation"]["positive_label"]
    intFolds = dctConfig["comparison"]["cv_folds"]
    dctPlots = dctConfig["plots"]

    # The same split every other stage uses.
    dfX, serY = tplBuildFeatures(dfData, dctConfig)
    dfXTrain, dfXTest, serYTrain, serYTest = lstSplitData(dfX, serY, dctConfig)

    # Fit each comparison model once for its curve, and cut its folds for the spread graph.
    dctProbabilities, dctFoldScores = {}, {}
    for strName, dctSettings in lstModelSettings(dctConfig):
        mdlModel = mdlBuildModel(dctSettings, intSeed)
        dctFoldScores[strName] = arrCrossValScores(mdlModel, dfXTrain, serYTrain, intFolds, intSeed)
        mdlModel.fit(dfXTrain, serYTrain)
        intPositive = list(mdlModel.classes_).index(varPositive)
        dctProbabilities[strName] = mdlModel.predict_proba(dfXTest)[:, intPositive]

    # The saved model drives the error counts and the group graph, as in evaluate.py.
    # The saved pipeline builds its own engineered columns, so it is given the raw test rows.
    dfRawTest = dfRawInputs(dfData, dctConfig).loc[dfXTest.index]
    arrPredictions = mdlSaved.predict(dfRawTest)
    intSaved = list(mdlSaved.classes_).index(varPositive)
    arrSavedProb = mdlSaved.predict_proba(dfRawTest)[:, intSaved]

    lstPaths = [
        pthSaveFigure(objPlotRocAndPr(dctProbabilities, serYTest, dctConfig), dctPlots["folder"], STR_ROC_FILE, dctPlots["dpi"]),
        pthSaveFigure(objPlotCvSpread(dctFoldScores, dctConfig), dctPlots["folder"], STR_CV_FILE, dctPlots["dpi"]),
        pthSaveFigure(objPlotConfusion(dfConfusionCounts(serYTest, arrPredictions, dctConfig)), dctPlots["folder"], STR_CONFUSION_FILE, dctPlots["dpi"]),
    ]

    # The group graph needs at least one grouping in the config.
    dfGroups = dfBuildGroups(dfData, dfXTest, dctConfig)
    if len(dfGroups.columns) > 0:
        dfByGroup = dfMetricsByGroup(dfGroups, serYTest, arrPredictions, arrSavedProb, dctConfig)
        lstPaths.append(pthSaveFigure(objPlotGroups(dfByGroup, dctConfig), dctPlots["folder"], STR_GROUPS_FILE, dctPlots["dpi"]))
    return lstPaths


def subMain() -> None:
    """Command line entry point: draw the evidence graphs from the saved model."""
    objParser = argparse.ArgumentParser(description="Draw the evidence graphs.")
    subAddConfigArgument(objParser)
    objParser.add_argument("--input", default=None, help="CSV to use (default: data.processed_path)")
    objParser.add_argument("--model", default=None, help="Saved model (default: model.path)")
    objArgs = objParser.parse_args()

    dctConfig = dctLoadConfig(objArgs.config)

    with objCaptureToRtf(dctConfig["outputs"]["rtf_path"], "plots"):
        if not dctConfig["plots"]["enabled"]:
            print("Graphs are switched off (plots.enabled is false in config.yaml).")
            return
        dfData = dfValidate(dfLoadCsv(objArgs.input or dctConfig["data"]["processed_path"]), dctConfig)
        mdlSaved = mdlLoadModel(objArgs.model or dctConfig["model"]["path"])
        lstPaths = lstDrawAllFigures(dfData, mdlSaved, dctConfig)
        print("Graphs written:")
        for pthFile in lstPaths:
            print(f"  {pthFile}")

        # The graphs are the evidence. Say what each one shows and what it means for use.
        subPrintStep(
            "Step 3 - Evaluate the models using at least one metric suited to the prediction task.",
            "",
        )


if __name__ == "__main__":
    subMain()
