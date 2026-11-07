"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    validate.py

    Stage 2 of the pipeline: data validation.

    Checks that the ingested data is fit to use before any feature building or
    training happens. Catching a bad file here, with a clear message, stops
    problems from turning into silent errors or odd model results further down.

    The checks are kept simple on purpose:
        1. The DataFrame is not empty.
        2. All required columns are present.
        3. There are at least validation.min_rows rows.
        4. The required columns contain no missing values.
        5. Columns with an allowed list (data.allowed_values) hold only those values.
        6. The value column (data.value_column) is numeric and inside min_value to max_value.
        7. There are no duplicate rows.
        8. The target column holds only 0 or 1.
        9. There are no columns beyond the ones data.keep_columns lists.
       10. The numeric feature columns hold numbers.
       11. Columns with limits in validation.column_ranges stay inside them.

    The required columns are the feature columns plus the target column from
    config.yaml, so the column names only ever need to be written in one place.

    The printed output is saved to an RTF file instead of the screen. The file name
    comes from outputs.rtf_path in config.yaml (the text {stage} becomes the stage name).

    Run from the project root:
        python -m src.validate
"""

import argparse

import pandas as pd

from src.config import subAddConfigArgument, dctLoadConfig
# Saves what the stage prints to an RTF file instead of the screen.
from src.display import (objCaptureToRtf, bDisplayDatasetDetails, subPrintPiplineHeader,
                         STR_BOLD, STR_UNDERLINE, STR_RESET, STR_ITALIC)
from src.ingest import dfLoadCsv


class ValidationError(Exception):
    """Raised when the data fails one or more checks.

    A custom exception type lets callers (and tests) catch data problems
    specifically, without also catching unrelated errors.
    """


def lstRequiredColumns(dctConfig: dict) -> list[str]:
    """Work out which columns the data must contain, from the config.

    The model needs its input columns and the target, so those are the columns
    that must be present and filled in.
    """
    dctFeatures = dctConfig["features"]
    return list(dctFeatures["feature_columns"]) + [dctFeatures["target_column"]]


def subReportCheck(strName: str, bPassed: bool, strDetail: str = "") -> None:
    """Print one check and its result, so the report shows every check that ran."""
    strResult = "PASS" if bPassed else "FAIL"
    strExtra = f" ({strDetail})" if strDetail else ""
    print(f"{STR_BOLD}{strName}:{STR_RESET} {strResult}{strExtra}")


def lstCheckData(dfData: pd.DataFrame, dctConfig: dict) -> list[str]:
    """Run every check, print the result of each and collect the problems found.

    This function never raises. It returns a list of plain English problems,
    which makes it easy to test and easy to report all issues at once. It also
    prints a PASS or FAIL line for each check, which goes to the stage's RTF file.

    Args:
        dfData: The DataFrame to check.
        dctConfig: The loaded config.yaml settings.

    Returns:
        A list of problem descriptions. An empty list means the data passed.
    """
    lstProblems: list[str] = []
    lstColumns = lstRequiredColumns(dctConfig)

    print(f"\n{STR_BOLD}{STR_UNDERLINE}Validation checks{STR_RESET}")

    # Check 1: an empty table cannot be trained on. Stop here, because the
    # later checks make no sense without any rows.
    if dfData.empty:
        subReportCheck("Check 1 - Table has rows", False, "DataFrame is empty")
        print("Remaining checks skipped.")
        return ["DataFrame is empty"]
    subReportCheck("Check 1 - Table has rows", True, f"{len(dfData)} rows")

    # Check 2: every required column must exist. Stop here if any are missing,
    # because the null check below would fail on a column that isn't there.
    lstMissing = [strColumn for strColumn in lstColumns if strColumn not in dfData.columns]
    if lstMissing:
        subReportCheck("Check 2 - Required columns present", False, f"missing {lstMissing}")
        print("Remaining checks skipped.")
        return [f"Missing columns: {lstMissing}"]
    subReportCheck("Check 2 - Required columns present", True, f"{len(lstColumns)} columns found")

    # Check 3: enough rows to train and test on. The limit is set in the config.
    intMinRows = dctConfig["validation"]["min_rows"]
    if len(dfData) < intMinRows:
        lstProblems.append(f"Only {len(dfData)} rows, need at least {intMinRows}")
        subReportCheck("Check 3 - Minimum rows", False, f"{len(dfData)} rows, need at least {intMinRows}")
    else:
        subReportCheck("Check 3 - Minimum rows", True, f"{len(dfData)} rows, minimum {intMinRows}")

    # Check 4: count missing values in each required column. isna() flags each
    # empty cell and sum() counts them per column.
    serNullCounts = dfData[lstColumns].isna().sum()
    lstNullProblems = [f"{strColumn} has {intCount} missing values"
                       for strColumn, intCount in serNullCounts.items() if intCount]
    lstProblems.extend(lstNullProblems)
    if lstNullProblems:
        subReportCheck("Check 4 - No missing values in required columns", False, "; ".join(lstNullProblems))
    else:
        subReportCheck("Check 4 - No missing values in required columns", True,
                       f"{len(lstColumns)} columns checked")

    # Check 5: columns with an allowed list hold only those values.
    dctAllowed = dctConfig["data"]["allowed_values"] or {}
    if not dctAllowed:
        subReportCheck("Check 5 - Allowed values", True, "no allowed lists in the config, nothing to check")
    for strColumn, lstAllowed in dctAllowed.items():
        if strColumn not in dfData.columns:
            lstProblems.append(f"{strColumn} is not in the data, so its allowed values cannot be checked")
            subReportCheck(f"Check 5 - Allowed values in {strColumn}", False, "column not in the data")
            continue
        intBad = int((~dfData[strColumn].isin(lstAllowed)).sum())
        if intBad:
            lstProblems.append(f"{strColumn} has {intBad} values outside {lstAllowed}")
            subReportCheck(f"Check 5 - Allowed values in {strColumn}", False,
                           f"{intBad} values outside {lstAllowed}")
        else:
            subReportCheck(f"Check 5 - Allowed values in {strColumn}", True,
                           f"0 values outside {lstAllowed}")

    # Check 6: the value column sits inside the configured range.
    strValue = dctConfig["data"]["value_column"]
    fltMin = dctConfig["data"]["min_value"]
    fltMax = dctConfig["data"]["max_value"]
    strRange = f"{fltMin if fltMin is not None else 'no limit'} to {fltMax if fltMax is not None else 'no limit'}"
    lstRangeProblems: list[str] = []
    if fltMin is None and fltMax is None:
        # No range is set in the config, so there is nothing to check.
        pass
    elif strValue not in dfData.columns:
        lstRangeProblems.append(f"{strValue} is not in the data")
    elif not pd.api.types.is_numeric_dtype(dfData[strValue]):
        lstRangeProblems.append(f"{strValue} is not numeric")
    else:
        if fltMin is not None and (dfData[strValue] < fltMin).any():
            lstRangeProblems.append(f"{strValue} has values below {fltMin}")
        if fltMax is not None and (dfData[strValue] > fltMax).any():
            lstRangeProblems.append(f"{strValue} has values above {fltMax}")
    lstProblems.extend(lstRangeProblems)
    if fltMin is None and fltMax is None:
        subReportCheck(f"Check 6 - {strValue} in range", True, "no range set in the config, nothing to check")
    elif lstRangeProblems:
        subReportCheck(f"Check 6 - {strValue} in range", False, "; ".join(lstRangeProblems))
    else:
        subReportCheck(f"Check 6 - {strValue} in range", True, f"range {strRange}")

    # Check 7: no duplicate rows.
    intDuplicates = int(dfData.duplicated().sum())
    if intDuplicates:
        lstProblems.append(f"{intDuplicates} duplicate rows")
        subReportCheck("Check 7 - No duplicate rows", False, f"{intDuplicates} duplicate rows")
    else:
        subReportCheck("Check 7 - No duplicate rows", True, "0 duplicate rows")

    # Check 8: the target is 0 or 1 only.
    strTarget = dctConfig["features"]["target_column"]
    intBadTarget = int((~dfData[strTarget].isin([0, 1])).sum())
    if intBadTarget:
        lstProblems.append(f"{strTarget} has {intBadTarget} values other than 0 and 1")
        subReportCheck(f"Check 8 - {strTarget} is 0 or 1", False, f"{intBadTarget} other values")
    else:
        subReportCheck(f"Check 8 - {strTarget} is 0 or 1", True, "0 other values")

    # Check 9: no unexpected columns. The expected set is the one ingest keeps.
    lstExpected = list(dctConfig["data"]["keep_columns"])
    lstExtra = [strColumn for strColumn in dfData.columns if strColumn not in lstExpected]
    if lstExtra:
        lstProblems.append(f"Unexpected columns: {lstExtra}")
        subReportCheck("Check 9 - No unexpected columns", False, f"unexpected {lstExtra}")
    else:
        subReportCheck("Check 9 - No unexpected columns", True, f"{len(dfData.columns)} columns, all expected")

    # Check 10: the feature columns that are not categories must hold numbers.
    lstCategorical = list((dctConfig["features"].get("categorical_columns") or {}).keys())
    lstNumeric = [strColumn for strColumn in dctConfig["features"]["feature_columns"] if strColumn not in lstCategorical]
    lstNotNumeric = [strColumn for strColumn in lstNumeric
                     if not pd.api.types.is_numeric_dtype(dfData[strColumn])]
    if lstNotNumeric:
        lstProblems.append(f"Not numeric: {lstNotNumeric}")
        subReportCheck("Check 10 - Numeric columns hold numbers", False, f"not numeric {lstNotNumeric}")
    else:
        subReportCheck("Check 10 - Numeric columns hold numbers", True, f"{len(lstNumeric)} columns checked")

    # Check 11: each column with limits in the config stays inside them.
    dctRanges = dctConfig["validation"]["column_ranges"] or {}
    if not dctRanges:
        subReportCheck("Check 11 - Column ranges", True, "no limits in the config, nothing to check")
    for strColumn, dctLimits in dctRanges.items():
        fltLow, fltHigh = dctLimits.get("min"), dctLimits.get("max")
        strLimits = f"{fltLow if fltLow is not None else 'no limit'} to {fltHigh if fltHigh is not None else 'no limit'}"
        if strColumn not in dfData.columns:
            lstProblems.append(f"{strColumn} is not in the data, so its range cannot be checked")
            subReportCheck(f"Check 11 - {strColumn} in range", False, "column not in the data")
        elif not pd.api.types.is_numeric_dtype(dfData[strColumn]):
            lstProblems.append(f"{strColumn} is not numeric, so its range cannot be checked")
            subReportCheck(f"Check 11 - {strColumn} in range", False, "column is not numeric")
        else:
            intLow = int((dfData[strColumn] < fltLow).sum()) if fltLow is not None else 0
            intHigh = int((dfData[strColumn] > fltHigh).sum()) if fltHigh is not None else 0
            if intLow or intHigh:
                lstProblems.append(f"{strColumn} has {intLow} values below {fltLow} and {intHigh} above {fltHigh}")
                subReportCheck(f"Check 11 - {strColumn} in range", False,
                               f"{intLow} below and {intHigh} above {strLimits}")
            else:
                subReportCheck(f"Check 11 - {strColumn} in range", True, f"range {strLimits}")

    return lstProblems


def dfValidate(dfData: pd.DataFrame, dctConfig: dict) -> pd.DataFrame:
    """Validate a DataFrame and return it unchanged if it passes.

    Returning the DataFrame lets other scripts chain the call, for example:
        dfData = dfValidate(dfLoadCsv(strPath), dctConfig)

    Raises:
        ValidationError: If any check fails. All problems are joined into one message.
    """
    lstProblems = lstCheckData(dfData, dctConfig)
    if lstProblems:
        raise ValidationError("; ".join(lstProblems))
    return dfData


def subMain() -> None:
    """Command line entry point: load the dataset and report whether it passes."""

 
    objParser = argparse.ArgumentParser(description="Validate the dataset.")
    subAddConfigArgument(objParser)
    objParser.add_argument("--input", default=None, help="CSV to check (default: data.processed_path)")
    objArgs = objParser.parse_args()

    dctConfig = dctLoadConfig(objArgs.config)
    strInputPath = objArgs.input or dctConfig["data"]["processed_path"]

    # Everything printed inside this block goes to the RTF file, not the screen.
    with objCaptureToRtf(dctConfig["outputs"]["rtf_path"], "Validate"):

        # Display the file header at the top of the stage output
        subPrintPiplineHeader(
            "Validate",
            dctConfig["project"]["client_question"],
            dctConfig["features"]["target_column"],
            ", ".join(dctConfig["features"]["feature_columns"]),
            strInputPath,
        )

        dfData = dfLoadCsv(strInputPath)
        bDisplayDatasetDetails(dfData) # Display the details for the dataset
        dfValidate(dfData, dctConfig)  # Raises ValidationError (and stops the script) on a failure.
        print(f"{len(dfData)} rows passed validation")


if __name__ == "__main__":
    subMain()
