"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    predict.py
    
    Stage 7 of the pipeline: prediction on new data.
    
    Loads the saved model and uses it to score a CSV of new rows. The new file only
    needs the feature columns. It does not need a target column, because the target
    is what we are predicting.
    
    The defaults come from config.yaml (data.new_data_path, model.path and
    outputs.predictions_path).
    
    The printed output is saved to an RTF file instead of the screen. The file name
    comes from outputs.rtf_path in config.yaml (the text {stage} becomes the stage name).

    Run from the project root:
        python -m src.predict
        python -m src.predict --input some/other/new_rows.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from src.config import subAddConfigArgument, dctLoadConfig
# Saves what the stage prints to an RTF file instead of the screen.
from src.display import (objCaptureToRtf, bDisplayDatasetDetails, subPrintPiplineHeader,
                         STR_BOLD, STR_UNDERLINE, STR_RESET, STR_ITALIC)
from src.approve import ModelNotApprovedError, strNotApprovedMessage, subExitNotApproved, subRequireSignoff
from src.evaluate import dctComputeMetrics
from src.features import dfRawInputs
from src.ingest import dfLoadCsv
from src.train import mdlLoadModel


def dfPredict(mdlModel, dfData: pd.DataFrame, dctConfig: dict) -> pd.DataFrame:
    """Add a prediction column to a DataFrame of new rows.

    Args:
        mdlModel: A fitted model.
        dfData: New data. Must contain every feature column named in the config.
        dctConfig: The loaded config.yaml settings.

    Returns:
        A copy of dfData with an extra column holding the predictions. Its
        name comes from outputs.prediction_column in the config.

    Raises:
        ValueError: If any feature column is missing from the input.
    """
    lstFeatureColumns = dctConfig["features"]["feature_columns"]

    # Check the columns first so the error names exactly what is missing,
    # instead of a vague scikit-learn error.
    lstMissing = [strColumn for strColumn in lstFeatureColumns if strColumn not in dfData.columns]
    if lstMissing:
        raise ValueError(f"Input is missing feature columns: {lstMissing}")

    # Work on a copy so the caller's DataFrame is not changed.
    dfOut = dfData.copy()

    # The name of the new column is a setting, so it can be changed in the config.
    strPredictionColumn = dctConfig["outputs"]["prediction_column"]

    # Pass only the raw feature columns. The saved pipeline builds the same
    # calculated and encoded columns that training used, so they cannot differ.
    dfOut[strPredictionColumn] = mdlModel.predict(dfRawInputs(dfData, dctConfig))
    return dfOut

def subPredictAll(objArgs, dctConfig: dict, strInputPath: str, strModelPath: str, strOutputPath: str) -> None:
    """Check the sign-off, score the CSV, save the predictions and score them against the kept-apart answers.

    Args:
        objArgs: The parsed command line. Uses input.
        dctConfig: The loaded config.yaml settings.
        strInputPath: The CSV to score.
        strModelPath: The saved model file.
        strOutputPath: Where to write the predictions.

    Returns:
        None. It prints, and writes the predictions file.

    Raises:
        ModelNotApprovedError: If no named person has approved the model. The message is printed first.
    """
    # Refuse to score patients with a model no named person has approved.
    try:
        subRequireSignoff(dctConfig, strModelPath)
    except ModelNotApprovedError as objError:
        print(strNotApprovedMessage(objError, dctConfig))
        raise
    mdlModel = mdlLoadModel(strModelPath)
    dfResult = dfPredict(mdlModel, dfLoadCsv(strInputPath), dctConfig)

    # Save the results, creating the folder if needed.
    pthOut = Path(strOutputPath)
    pthOut.parent.mkdir(parents=True, exist_ok=True)
    dfResult.to_csv(pthOut, index=False)
    print(f"\nWrote {len(dfResult)} predictions to {pthOut}")

    # When the input is the set-aside new patients file, their real answers are
    # kept apart. Compare them with the predictions so the run shows how the
    # model does on patients it has never seen. Not done when --input is used,
    # because those rows would not line up with the answers file.
    pthAnswers = Path(dctConfig["data"]["new_data_answers_path"])
    if objArgs.input is None and pthAnswers.is_file():
        serAnswers = pd.read_csv(pthAnswers)[dctConfig["features"]["target_column"]]
        if len(serAnswers) == len(dfResult):
            dctNewScores = dctComputeMetrics(serAnswers, dfResult[dctConfig["outputs"]["prediction_column"]], dctConfig)
            print(f"\n{STR_BOLD}SCORES ON {len(dfResult)} NEW PATIENTS (answers kept apart from the model){STR_RESET}")
            print(f"\n{dctNewScores}")
        else:
            print(f"\nThe answers file has a different number of rows from the new patients file, so it was not used.")






def subMain() -> None:
    """Command line entry point: score a CSV and write the predictions."""
    objParser = argparse.ArgumentParser(description="Predict on new data.")
    subAddConfigArgument(objParser)
    objParser.add_argument("--input", default=None, help="CSV to score (default: data.new_data_path)")
    objParser.add_argument("--model", default=None, help="Saved model (default: model.path)")
    objParser.add_argument("--output", default=None, help="Results file (default: outputs.predictions_path)")
    objArgs = objParser.parse_args()

    dctConfig = dctLoadConfig(objArgs.config)
    strInputPath = objArgs.input or dctConfig["data"]["new_data_path"]
    strModelPath = objArgs.model or dctConfig["model"]["path"]
    strOutputPath = objArgs.output or dctConfig["outputs"]["predictions_path"]

    # Everything printed inside this block goes to the RTF file, not the screen.
    try:
        with objCaptureToRtf(dctConfig["outputs"]["rtf_path"], "Predict"):
            # Display the file header at the top of the stage output
            subPrintPiplineHeader(
                "Predict",
                dctConfig["project"]["client_question"],
                dctConfig["features"]["target_column"],
                ", ".join(dctConfig["features"]["feature_columns"]),
                strInputPath,
            )

            subPredictAll(objArgs, dctConfig, strInputPath, strModelPath, strOutputPath)

    except ModelNotApprovedError as objError:
        # The RTF file already holds the message. Show it on the screen too and stop cleanly.
        subExitNotApproved(objError, dctConfig)



if __name__ == "__main__":
    subMain()
