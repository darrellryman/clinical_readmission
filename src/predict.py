"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    predict.py
    
    Stage 6 of the pipeline: prediction on new data.
    
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
from src.display import (objCaptureToRtf, bDisplayDatasetDetails, subPrintStep,
                         STR_BOLD, STR_UNDERLINE, STR_RESET, STR_ITALIC)
from src.evaluate import dctComputeMetrics
from src.features import dfEngineerFeatures
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

    # Select the feature columns, then build the same calculated and encoded
    # columns that training used. Skipping this step would feed the model
    # different columns from the ones it learned on.
    dfX = dfEngineerFeatures(dfData[lstFeatureColumns], dctConfig)
    dfOut[strPredictionColumn] = mdlModel.predict(dfX)
    return dfOut


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
    with objCaptureToRtf(dctConfig["outputs"]["rtf_path"], "predict"):
        mdlModel = mdlLoadModel(strModelPath)
        dfResult = dfPredict(mdlModel, dfLoadCsv(strInputPath), dctConfig)

        # Save the results, creating the folder if needed.
        pthOut = Path(strOutputPath)
        pthOut.parent.mkdir(parents=True, exist_ok=True)
        dfResult.to_csv(pthOut, index=False)
        print(f"Wrote {len(dfResult)} predictions to {pthOut}")

        # When the input is the set-aside new patients file, their real answers are
        # kept apart. Compare them with the predictions so the run shows how the
        # model does on patients it has never seen. Not done when --input is used,
        # because those rows would not line up with the answers file.
        pthAnswers = Path(dctConfig["data"]["new_data_answers_path"])
        if objArgs.input is None and pthAnswers.is_file():
            serAnswers = pd.read_csv(pthAnswers)[dctConfig["features"]["target_column"]]
            if len(serAnswers) == len(dfResult):
                dctNewScores = dctComputeMetrics(serAnswers, dfResult[dctConfig["outputs"]["prediction_column"]], dctConfig)
                print(f"\nSCORES ON {len(dfResult)} NEW PATIENTS (answers kept apart from the model): {dctNewScores}")
            else:
                print("The answers file has a different number of rows from the new patients file, so it was not used.")

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


if __name__ == "__main__":
    subMain()
