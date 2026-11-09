"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    monitor.py

    Monitoring check for a new batch of patients. It is run by hand, like predict.py,
    because it needs a batch that you supply. It is not a DVC stage.

    It compares the batch with the cleaned training data (the reference) and reports
    PASS, WATCH or ALERT for three things:
        1. Input drift. The population stability index (PSI) of each input column,
           batch against training. Under 0.1 is stable, 0.1 to 0.25 is a shift worth
           watching, over 0.25 is a large shift.
        2. Data quality. The share of values that are missing, outside the limits in
           the config, or in a category the model has never seen.
        3. Output drift. The share of the batch the model flags as likely readmissions,
           against the share it flags in the training data.

    The thresholds come from the monitoring section of config.yaml. Nothing is
    trained, saved or overwritten here. The model is only used to count its flags.

    The printed output is saved to an RTF file instead of the screen. The file name
    comes from outputs.rtf_path in config.yaml (the text {stage} becomes the stage name).
    The exit code is 1 when any check is an ALERT, so a scheduler or CI job can act on it.

    Run from the project root, after train:
        python -m src.monitor
        python -m src.monitor --input some/other/new_rows.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd

from src.config import subAddConfigArgument, dctLoadConfig
from src.display import (objCaptureToRtf, subPrintPiplineHeader,
                         STR_BOLD, STR_UNDERLINE, STR_RESET, STR_ITALIC)
from src.features import dfRawInputs
from src.ingest import dfLoadCsv
from src.train import mdlLoadModel

# The three results, from best to worst. The position in this list ranks them.
LST_STATUS = ["PASS", "WATCH", "ALERT"]

# A tiny share stands in for zero, so a category or bin that is empty in one
# sample can't make the PSI log or division blow up.
FLT_EPSILON = 0.0001


def strStatus(fltValue: float, fltWatch: float, fltAlert: float) -> str:
    """Turn a measured value into PASS, WATCH or ALERT.

    Args:
        fltValue: The measurement. Bigger is worse.
        fltWatch: The value from which the result is WATCH.
        fltAlert: The value from which the result is ALERT.

    Returns:
        "PASS", "WATCH" or "ALERT".
    """
    if fltValue >= fltAlert:
        return "ALERT"
    if fltValue >= fltWatch:
        return "WATCH"
    return "PASS"


def strWorst(lstStatuses: list) -> str:
    """Return the worst result in a list, or PASS if the list is empty."""
    return max(lstStatuses, key=LST_STATUS.index) if lstStatuses else "PASS"


def fltPsiFromShares(arrExpected: np.ndarray, arrActual: np.ndarray) -> float:
    """Population stability index from two lists of shares that each add up to 1.

    Args:
        arrExpected: The share of the reference in each bin or category.
        arrActual: The share of the batch in the same bins or categories.

    Returns:
        The PSI. 0 means identical. Larger means a bigger shift.
    """
    arrE = np.clip(np.asarray(arrExpected, dtype=float), FLT_EPSILON, None)
    arrA = np.clip(np.asarray(arrActual, dtype=float), FLT_EPSILON, None)
    return float(np.sum((arrA - arrE) * np.log(arrA / arrE)))


def fltNumericPsi(serReference: pd.Series, serBatch: pd.Series, intBins: int) -> float:
    """PSI for a numeric column. The bins are the reference's own quantiles.

    Values in the batch beyond the reference's lowest or highest value fall into the
    first or last bin, so a batch that has moved far away still shows as a shift.

    Args:
        serReference: The column in the training data.
        serBatch: The same column in the new batch.
        intBins: The number of bins to aim for. Fewer are used if the column has few distinct values.

    Returns:
        The PSI.
    """
    arrEdges = np.unique(np.quantile(serReference.dropna(), np.linspace(0, 1, intBins + 1)))
    if len(arrEdges) < 2:
        # A column that never changes in training can only be compared as "same value or not".
        fltSame = float((serBatch.dropna() == arrEdges[0]).mean()) if len(serBatch.dropna()) else 0.0
        return fltPsiFromShares([1.0, 0.0], [fltSame, 1.0 - fltSame])
    # Open both ends, then count how many values land in each bin.
    arrEdges[0], arrEdges[-1] = -np.inf, np.inf
    arrRef = np.histogram(serReference.dropna(), bins=arrEdges)[0] / max(len(serReference.dropna()), 1)
    arrNew = np.histogram(serBatch.dropna(), bins=arrEdges)[0] / max(len(serBatch.dropna()), 1)
    return fltPsiFromShares(arrRef, arrNew)


def fltCategoryPsi(serReference: pd.Series, serBatch: pd.Series) -> float:
    """PSI for a text column, using every category seen in either sample."""
    lstCategories = sorted(set(serReference.dropna().unique()) | set(serBatch.dropna().unique()), key=str)
    serRef = serReference.value_counts(normalize=True)
    serNew = serBatch.value_counts(normalize=True)
    return fltPsiFromShares([serRef.get(c, 0.0) for c in lstCategories], [serNew.get(c, 0.0) for c in lstCategories])


def dfInputDrift(dfReference: pd.DataFrame, dfBatch: pd.DataFrame, dctConfig: dict) -> pd.DataFrame:
    """Work out the PSI of every input column and give each a result.

    Args:
        dfReference: The cleaned training data.
        dfBatch: The new batch.
        dctConfig: The loaded config.yaml settings.

    Returns:
        One row per input column with its PSI and result.
    """
    dctMon = dctConfig["monitoring"]
    lstRows = []
    for strColumn in dctConfig["features"]["feature_columns"]:
        if pd.api.types.is_numeric_dtype(dfReference[strColumn]) and pd.api.types.is_numeric_dtype(dfBatch[strColumn]):
            fltPsi = fltNumericPsi(dfReference[strColumn], dfBatch[strColumn], dctMon["psi_bins"])
        else:
            fltPsi = fltCategoryPsi(dfReference[strColumn], dfBatch[strColumn])
        lstRows.append({"column": strColumn, "psi": round(fltPsi, 4),
                        "result": strStatus(fltPsi, dctMon["psi_watch"], dctMon["psi_alert"])})
    return pd.DataFrame(lstRows)


def dfOutOfRange(dfBatch: pd.DataFrame, dctConfig: dict) -> pd.DataFrame:
    """Count the values outside the configured limits and the unseen categories.

    The limits are the ones validate.py uses: validation.column_ranges, the filter on
    data.value_column, and the categories in features.categorical_columns.

    Args:
        dfBatch: The new batch.
        dctConfig: The loaded config.yaml settings.

    Returns:
        One row per column with a limit or category list: the count and share outside it.
    """
    dctLimits = {k: dict(v) for k, v in (dctConfig["validation"]["column_ranges"] or {}).items()}
    dctData = dctConfig["data"]
    strValueColumn = dctData.get("value_column")
    if strValueColumn and (dctData.get("min_value") is not None or dctData.get("max_value") is not None):
        dctLimits[strValueColumn] = {"min": dctData.get("min_value"), "max": dctData.get("max_value")}

    lstRows = []
    for strColumn, dctLimit in dctLimits.items():
        if strColumn not in dfBatch.columns:
            continue
        serValues = pd.to_numeric(dfBatch[strColumn], errors="coerce").dropna()
        intBad = 0
        if dctLimit.get("min") is not None:
            intBad += int((serValues < dctLimit["min"]).sum())
        if dctLimit.get("max") is not None:
            intBad += int((serValues > dctLimit["max"]).sum())
        lstRows.append({"column": strColumn, "check": "outside limits", "count": intBad,
                        "share": intBad / max(len(dfBatch), 1)})
    for strColumn, lstAllowed in (dctConfig["features"]["categorical_columns"] or {}).items():
        if strColumn not in dfBatch.columns:
            continue
        intBad = int((~dfBatch[strColumn].dropna().isin(lstAllowed)).sum())
        lstRows.append({"column": strColumn, "check": "unseen category", "count": intBad,
                        "share": intBad / max(len(dfBatch), 1)})
    return pd.DataFrame(lstRows, columns=["column", "check", "count", "share"])


def dfDataQuality(dfBatch: pd.DataFrame, dctConfig: dict) -> pd.DataFrame:
    """Missing values, values outside the limits and unseen categories, each with a result.

    Args:
        dfBatch: The new batch.
        dctConfig: The loaded config.yaml settings.

    Returns:
        One row per check with its count, share and result.
    """
    dctMon = dctConfig["monitoring"]
    lstRows = []
    for strColumn in dctConfig["features"]["feature_columns"]:
        intMissing = int(dfBatch[strColumn].isna().sum())
        fltShare = intMissing / max(len(dfBatch), 1)
        lstRows.append({"column": strColumn, "check": "missing", "count": intMissing, "share": round(fltShare, 4),
                        "result": strStatus(fltShare, dctMon["bad_share_watch"], dctMon["bad_share_alert"])})
    for _, dfRow in dfOutOfRange(dfBatch, dctConfig).iterrows():
        lstRows.append({"column": dfRow["column"], "check": dfRow["check"], "count": int(dfRow["count"]),
                        "share": round(float(dfRow["share"]), 4),
                        "result": strStatus(float(dfRow["share"]), dctMon["bad_share_watch"], dctMon["bad_share_alert"])})
    return pd.DataFrame(lstRows)


def dfUsableRows(dfData: pd.DataFrame, dctConfig: dict) -> pd.DataFrame:
    """Keep the rows the model can score: no missing inputs and no unseen categories.

    Args:
        dfData: A batch with every feature column present.
        dctConfig: The loaded config.yaml settings.

    Returns:
        The usable rows only. Quality problems in the dropped rows are reported by dfDataQuality.
    """
    dfOut = dfData.dropna(subset=dctConfig["features"]["feature_columns"])
    for strColumn, lstAllowed in (dctConfig["features"]["categorical_columns"] or {}).items():
        dfOut = dfOut[dfOut[strColumn].isin(lstAllowed)]
    return dfOut


def fltFlagRate(mdlModel, dfData: pd.DataFrame, dctConfig: dict) -> float:
    """The share of rows the model flags as likely readmissions."""
    if len(dfData) == 0:
        return float("nan")
    return float(np.mean(mdlModel.predict(dfRawInputs(dfData, dctConfig))))


def tplOutputDrift(mdlModel, dfReference: pd.DataFrame, dfBatch: pd.DataFrame, dctConfig: dict) -> tuple:
    """Compare the model's flag rate on the batch with its flag rate on the training data.

    Args:
        mdlModel: The saved model.
        dfReference: The cleaned training data.
        dfBatch: The new batch.
        dctConfig: The loaded config.yaml settings.

    Returns:
        A tuple of the reference flag rate, the batch flag rate, the batch rows scored and the result.
        The result is ALERT if no row in the batch could be scored.
    """
    dctMon = dctConfig["monitoring"]
    dfUsable = dfUsableRows(dfBatch, dctConfig)
    fltReference = fltFlagRate(mdlModel, dfUsableRows(dfReference, dctConfig), dctConfig)
    fltBatch = fltFlagRate(mdlModel, dfUsable, dctConfig)
    if np.isnan(fltBatch):
        return fltReference, fltBatch, 0, "ALERT"
    return fltReference, fltBatch, len(dfUsable), strStatus(abs(fltBatch - fltReference), dctMon["flag_rate_watch"], dctMon["flag_rate_alert"])


def subMonitorAll(dctConfig: dict, strInputPath: str, strModelPath: str) -> str:
    """Run every check, print the results and return the worst one.

    Args:
        dctConfig: The loaded config.yaml settings.
        strInputPath: The batch of new patients (a CSV with the feature columns).
        strModelPath: The saved model file.

    Returns:
        The worst result across all checks: PASS, WATCH or ALERT.

    Raises:
        ValueError: If the batch is missing any feature column.
    """
    dfBatch = dfLoadCsv(strInputPath)
    dfReference = dfLoadCsv(dctConfig["data"]["processed_path"])
    lstMissing = [c for c in dctConfig["features"]["feature_columns"] if c not in dfBatch.columns]
    if lstMissing:
        raise ValueError(f"Input is missing feature columns: {lstMissing}")
    dctMon = dctConfig["monitoring"]

    print(f"\n{STR_BOLD}{STR_UNDERLINE}Batch{STR_RESET}")
    print(f"{len(dfBatch)} new rows from {strInputPath}, checked against {len(dfReference)} training rows.")
    blnSmall = len(dfBatch) < dctMon["min_batch_rows"]
    if blnSmall:
        print(f"{STR_ITALIC}The batch has fewer than {dctMon['min_batch_rows']} rows, so the drift numbers are noisy. Read them with care.{STR_RESET}")

    print(f"\n{STR_BOLD}{STR_UNDERLINE}1. Input drift (PSI, batch against training){STR_RESET}")
    print(f"WATCH from {dctMon['psi_watch']}, ALERT from {dctMon['psi_alert']}.\n")
    dfDrift = dfInputDrift(dfReference, dfBatch, dctConfig)
    print(dfDrift.to_string(index=False))

    print(f"\n{STR_BOLD}{STR_UNDERLINE}2. Data quality{STR_RESET}")
    print(f"WATCH from {dctMon['bad_share_watch']:.0%} of rows, ALERT from {dctMon['bad_share_alert']:.0%}.\n")
    dfQuality = dfDataQuality(dfBatch, dctConfig)
    print(dfQuality.to_string(index=False))

    print(f"\n{STR_BOLD}{STR_UNDERLINE}3. Output drift (share of patients flagged){STR_RESET}")
    mdlModel = mdlLoadModel(strModelPath)
    fltRef, fltNew, intScored, strFlagResult = tplOutputDrift(mdlModel, dfReference, dfBatch, dctConfig)
    if intScored == 0:
        print("No row in the batch could be scored (missing inputs or unseen categories).")
    else:
        print(f"Training data: {fltRef:.1%} flagged. Batch: {fltNew:.1%} flagged ({intScored} rows scored). "
              f"Change: {abs(fltNew - fltRef) * 100:.1f} percentage points. WATCH from {dctMon['flag_rate_watch']:.0%}, ALERT from {dctMon['flag_rate_alert']:.0%}.")
    print(f"Result: {strFlagResult}")

    strOverall = strWorst(list(dfDrift["result"]) + list(dfQuality["result"]) + [strFlagResult])
    print(f"\n{STR_BOLD}{STR_UNDERLINE}Overall{STR_RESET}")
    print(f"{STR_BOLD}{strOverall}{STR_RESET}. Input drift: {strWorst(list(dfDrift['result']))}. "
          f"Data quality: {strWorst(list(dfQuality['result']))}. Output drift: {strFlagResult}.")
    if strOverall == "ALERT":
        print("An ALERT means do not trust predictions from this batch until someone has looked at the data.")
    print("Recall can only be checked once the real outcomes arrive, so it is not part of this check.")
    return strOverall


def subMain() -> None:
    """Command line entry point: check a batch of new patients and report any drift."""
    objParser = argparse.ArgumentParser(description="Check a batch of new patients for drift and bad data.")
    subAddConfigArgument(objParser)
    objParser.add_argument("--input", default=None, help="CSV to check (default: data.new_data_path)")
    objParser.add_argument("--model", default=None, help="Saved model (default: model.path)")
    objArgs = objParser.parse_args()

    dctConfig = dctLoadConfig(objArgs.config)
    strInputPath = objArgs.input or dctConfig["data"]["new_data_path"]
    strModelPath = objArgs.model or dctConfig["model"]["path"]

    # Everything printed inside this block goes to the RTF file, not the screen.
    with objCaptureToRtf(dctConfig["outputs"]["rtf_path"], "Monitor"):
        subPrintPiplineHeader(
            "Monitor",
            dctConfig["project"]["client_question"],
            dctConfig["features"]["target_column"],
            ", ".join(dctConfig["features"]["feature_columns"]),
            strInputPath,
        )
        strOverall = subMonitorAll(dctConfig, strInputPath, strModelPath)

    # A non-zero exit lets a scheduler or CI job notice an ALERT.
    if strOverall == "ALERT":
        sys.exit(1)


if __name__ == "__main__":
    subMain()
