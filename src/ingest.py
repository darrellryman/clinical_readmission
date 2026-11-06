"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    ingest.py

    Stage 1 of the pipeline: data ingestion.

    This module reads the raw data from disk into a pandas DataFrame, keeps only the
    columns listed in config.yaml, cleans it, and writes the cleaned copy back out.
    The raw file is never changed. It also prints a profile of the data before and
    after, so you can see what the cleaning did. Checking the cleaned data against
    the project's rules is the next stage (validate.py).

    The cleaning steps are: tidy the column names (trim spaces, lower case), trim
    spaces around text values, remove duplicate rows, and remove rows whose value in
    a column is not on that column's allowed list (data.allowed_values), and remove
    rows outside the range data.min_value to data.max_value in data.value_column.

    The file locations and the columns to keep come from config.yaml (data.raw_path,
    data.processed_path and data.keep_columns). Later stages read the processed file.

    Where the raw data comes from is set by data.source. "file" reads data.raw_path.
    "url" downloads data.raw_url into data.raw_path (with a timeout and retries)
    and reads that copy. Each run prints the source, the location, the time and a
    fingerprint of the file, so you can show which data a run used.

    The printed output is saved to an RTF file instead of the screen. The file name
    comes from outputs.rtf_path in config.yaml (the text {stage} becomes the stage name).

    Run from the project root:
        python -m src.ingest
        python -m src.ingest --input some/other/file.csv     (overrides the config)
"""

# argparse gives the script a small command line interface (--input, --output, --help).
import argparse

# datetime stamps the load; hashlib fingerprints the file; time and urllib.request
# are used to download a url source with a pause between attempts.
import datetime
import hashlib
import time
import urllib.error
import urllib.request

# Path handles file paths the same way on Windows, macOS and Linux.
from pathlib import Path

# pandas reads the CSV and holds the data as a table (DataFrame).
import pandas as pd

# train_test_split cuts a table into two random pieces. Used here to set aside new patients.
from sklearn.model_selection import train_test_split

# The shared config loader, and the helper that adds the --config option.
from src.config import subAddConfigArgument, dctLoadConfig

# The shared function that prints a profile of the data.
from src.display import (objCaptureToRtf, bDisplayDatasetDetails, subPrintStep, subPrintPiplineHeader,
                         STR_BOLD, STR_UNDERLINE, STR_RESET, STR_ITALIC)

def dfLoadCsv(varPath) -> pd.DataFrame:
    """Read a CSV file into a DataFrame.

    Args:
        varPath: Location of the CSV file, as text or a Path. Callers get this
            from config.yaml.

    Returns:
        The file contents as a DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist. Failing here, with a
            clear message, is better than a confusing pandas error later.
        ValueError: If the file is empty or has a header but no rows.
    """
    # Accept either a string or a Path from the caller.
    pthFile = Path(varPath)

    # Check up front so the error message names the exact file that is missing.
    if not pthFile.exists():
        raise FileNotFoundError(f"Input file not found: {pthFile}")

    # An empty file would otherwise fail later with a pandas message that doesn't name the file.
    try:
        dfData = pd.read_csv(pthFile)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Input file is empty: {pthFile}") from None
    if len(dfData) == 0:
        raise ValueError(f"Input file has a header but no rows: {pthFile}")
    return dfData


def pthDownloadFile(strUrl: str, varPath, intTimeout: int, intRetries: int) -> Path:
    """Download a web address to a local file, trying again if it fails.

    The file is written to a temporary name and renamed only when the whole
    download worked, so a failed or half finished download never replaces a
    good local copy.

    Args:
        strUrl: The http or https address to download.
        varPath: Where to save the file, as text or a Path. The folder is
            created if it doesn't exist.
        intTimeout: Seconds to wait for the server on each attempt.
        intRetries: How many attempts to make in total (at least 1).

    Returns:
        The path of the saved file.

    Raises:
        ValueError: If the address isn't http or https. This also stops file:
            and ftp: addresses from being used by mistake.
        ConnectionError: If every attempt failed, or the download was empty.
            The message names the address and the last error.
    """
    if not str(strUrl).lower().startswith(("http://", "https://")):
        raise ValueError(f"Only http and https addresses are allowed: {strUrl}")

    pthFile = Path(varPath)
    pthFile.parent.mkdir(parents=True, exist_ok=True)
    pthTemp = pthFile.with_name(pthFile.name + ".part")
    strLastError = ""

    for intAttempt in range(1, intRetries + 1):
        try:
            with urllib.request.urlopen(strUrl, timeout=intTimeout) as objResponse:
                bytBody = objResponse.read()
            if not bytBody:
                raise ConnectionError("the server returned an empty file")
            pthTemp.write_bytes(bytBody)
            pthTemp.replace(pthFile)
            return pthFile
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as objError:
            strLastError = str(objError)
            print(f"Download attempt {intAttempt} of {intRetries} failed: {strLastError}")
            # Wait a little longer after each failure (1 s, 2 s, 3 s ...) before trying again.
            if intAttempt < intRetries:
                time.sleep(intAttempt)

    raise ConnectionError(f"Could not download {strUrl} after {intRetries} attempts. Last error: {strLastError}")


def strFileFingerprint(varPath) -> str:
    """Return a short fingerprint of a file, so two loads can be compared.

    Args:
        varPath: The file to fingerprint, as text or a Path.

    Returns:
        The first 12 characters of the file's SHA-256 hash. The same contents
        always give the same fingerprint.
    """
    objHash = hashlib.sha256()
    with Path(varPath).open("rb") as objHandle:
        for bytChunk in iter(lambda: objHandle.read(1024 * 1024), b""):
            objHash.update(bytChunk)
    return objHash.hexdigest()[:12]


def tplLoadRawData(dctData: dict) -> tuple:
    """Load the raw data from the source named in config.yaml, the same way every run.

    data.source decides where the data comes from. For "file" the CSV at
    data.raw_path is read. For "url" data.raw_url is downloaded into
    data.raw_path first and that local copy is read, so the rest of the
    pipeline always works from one raw file in one place.

    Args:
        dctData: The data section of the config.

    Returns:
        A tuple of the DataFrame and a dictionary that describes the load:
        source, location, local_file, loaded_at and fingerprint. Print the
        dictionary so the run leaves evidence of what was loaded.

    Raises:
        FileNotFoundError, ValueError, ConnectionError: See dfLoadCsv and pthDownloadFile.
    """
    strSource = dctData["source"]
    pthLocal = Path(dctData["raw_path"])

    if strSource == "url":
        strLocation = dctData["raw_url"]
        pthDownloadFile(strLocation, pthLocal, dctData["download_timeout"], dctData["download_retries"])
    else:
        strLocation = str(pthLocal)

    dfData = dfLoadCsv(pthLocal)
    dctInfo = {
        "source": strSource,
        "location": strLocation,
        "local_file": str(pthLocal),
        "loaded_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fingerprint": strFileFingerprint(pthLocal),
    }
    return dfData, dctInfo


def dfKeepColumns(dfData: pd.DataFrame, lstKeep: list) -> pd.DataFrame:
    """Return a copy of the data with only the listed columns, in that order.

    Args:
        dfData: The raw data.
        lstKeep: Column names to keep, from data.keep_columns in config.yaml.

    Returns:
        A new DataFrame holding only those columns.

    Raises:
        KeyError: If any listed column is not in the data. The message names
            every missing column, so a typo is easy to spot.
    """
    # Find every requested column that the file doesn't have.
    lstMissing = [strCol for strCol in lstKeep if strCol not in dfData.columns]
    if lstMissing:
        raise KeyError(f"Columns not found in the data: {lstMissing}")

    # .copy() so later changes to the result can't touch the original.
    return dfData[lstKeep].copy()


def dfCleanData(dfData: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the data. The original DataFrame is not changed.

    Steps:
        1. Trim spaces from the column names and make them lower case, so
           " Age" and "age" can't end up as two different columns.
        2. Trim spaces from the start and end of every text value, so "yes "
           and "yes" count as the same value.
        3. Remove rows that are exact duplicates of an earlier row. The first
           copy is kept.
        4. Renumber the rows from 0, so the index has no gaps.

    Args:
        dfData: The data to clean.

    Returns:
        A new, cleaned DataFrame.
    """
    # Work on a copy so the caller's table is left as it was.
    dfClean = dfData.copy()

    # Step 1: tidy the column names.
    dfClean.columns = dfClean.columns.str.strip().str.lower()

    # Step 2: trim text values. Number columns are skipped.
    for strCol in dfClean.columns:
        if pd.api.types.is_string_dtype(dfClean[strCol]):
            dfClean[strCol] = dfClean[strCol].str.strip()

    # Steps 3 and 4: drop exact duplicates, then renumber the rows.
    return dfClean.drop_duplicates().reset_index(drop=True)


def dfFilterAllowedValues(dfData: pd.DataFrame, dctAllowed: dict) -> pd.DataFrame:
    """Keep only the rows whose values are on the allowed list for each column.

    Args:
        dfData: The data to filter.
        dctAllowed: Column name -> list of allowed values, from
            data.allowed_values in config.yaml. An empty dictionary keeps
            every row.

    Returns:
        A new DataFrame without the rows that have any other value, blanks
        included. The rows are renumbered from 0.

    Raises:
        KeyError: If a listed column is not in the data. The message names
            every missing column.
    """
    # Find every column in the rules that the data doesn't have.
    lstMissing = [strCol for strCol in dctAllowed if strCol not in dfData.columns]
    if lstMissing:
        raise KeyError(f"allowed_values names columns not in the data: {lstMissing}")

    # Apply each rule in turn. isin() is True where the value is on the list.
    dfOut = dfData
    for strCol, lstAllowed in dctAllowed.items():
        dfOut = dfOut[dfOut[strCol].isin(lstAllowed)]

    # .copy() so later changes can't touch the original, and renumber the rows.
    return dfOut.reset_index(drop=True).copy()


def dfFilterValueRange(dfData: pd.DataFrame, strValueColumn: str, fltMinValue, fltMaxValue) -> pd.DataFrame:
    """Keep only the rows where a numeric column falls inside a range, ends included.

    Args:
        dfData: The data to filter.
        strValueColumn: The column to filter on, from data.value_column in config.yaml.
        fltMinValue: The lowest value to keep, from data.min_value. None leaves
            the low end open.
        fltMaxValue: The highest value to keep, from data.max_value. None leaves
            the high end open. Setting both keeps one band of values.

    Returns:
        A new DataFrame without the rows outside the range, blanks included.
        The rows are renumbered from 0.

    Raises:
        KeyError: If the column is not in the data.
        TypeError: If the column does not hold numbers.
    """
    if fltMinValue is None and fltMaxValue is None:
        return dfData.reset_index(drop=True).copy()

    if strValueColumn not in dfData.columns:
        raise KeyError(f"value_column '{strValueColumn}' is not in the data")
    if not pd.api.types.is_numeric_dtype(dfData[strValueColumn]):
        raise TypeError(f"value_column '{strValueColumn}' must hold numbers to apply min_value and max_value")

    # Comparisons are False for a blank, so a row with no value is removed too.
    serValue = dfData[strValueColumn]
    serKeep = pd.Series(True, index=dfData.index)
    if fltMinValue is not None:
        serKeep = serKeep & (serValue >= fltMinValue)
    if fltMaxValue is not None:
        serKeep = serKeep & (serValue <= fltMaxValue)
    return dfData[serKeep].reset_index(drop=True).copy()


def pthSaveCsv(dfData: pd.DataFrame, varPath) -> Path:
    """Write a DataFrame to a CSV file, creating the folder if needed.

    Args:
        dfData: The data to save.
        varPath: Where to write the file, as text or a Path.

    Returns:
        The path that was written, so the caller can report it.
    """
    pthFile = Path(varPath)

    # Make sure the destination folder exists (e.g. data/processed/).
    # parents=True creates any missing parent folders, exist_ok=True stops an
    # error if the folder is already there.
    pthFile.parent.mkdir(parents=True, exist_ok=True)

    # index=False keeps pandas from adding an extra row-number column.
    dfData.to_csv(pthFile, index=False)
    return pthFile


def tplHoldOutNewPatients(dfData: pd.DataFrame, dctConfig: dict) -> tuple:
    """Set aside a share of patients to act as new patients the model never sees.

    This happens before any training or testing, and the random choice uses the
    seed from config.yaml, so the same patients are set aside every run. The
    split is stratified on the target, so the readmission rate in the set-aside
    group matches the rest.

    Args:
        dfData: The cleaned data, with the target column still in it.
        dctConfig: The loaded config.yaml settings. Uses data.new_data_fraction,
            features.target_column and project.random_state.

    Returns:
        A tuple of two DataFrames, both with a fresh 0, 1, 2... index: the rows
        kept for training and testing, and the rows set aside. If
        data.new_data_fraction is 0, nothing is set aside and the second
        DataFrame is empty.

    Raises:
        KeyError: If the target column is not in dfData.
    """
    fltFraction = dctConfig["data"]["new_data_fraction"]
    if fltFraction == 0:
        return dfData.reset_index(drop=True), dfData.iloc[0:0].reset_index(drop=True)

    # Stratify on the target so both groups keep the same readmission rate.
    dfKept, dfNew = train_test_split(
        dfData,
        test_size=fltFraction,
        stratify=dfData[dctConfig["features"]["target_column"]],
        random_state=dctConfig["project"]["random_state"],
    )
    return dfKept.reset_index(drop=True), dfNew.reset_index(drop=True)


def tplSaveNewPatients(dfNew: pd.DataFrame, dctConfig: dict) -> tuple:
    """Write the set-aside patients to two files: the questions and the answers.

    Args:
        dfNew: The set-aside patients, from tplHoldOutNewPatients(), including
            the target column.
        dctConfig: The loaded config.yaml settings. Uses features.target_column,
            data.new_data_path and data.new_data_answers_path.

    Returns:
        A tuple of the two paths written: the new-patients file (no answers) and
        the answers file. Both are None when dfNew is empty and nothing was written.
    """
    if len(dfNew) == 0:
        return None, None

    strTarget = dctConfig["features"]["target_column"]

    # The model must never see the answer, so it is split off into its own file.
    # Both files keep the same row order, so row 1 matches row 1.
    pthNew = pthSaveCsv(dfNew.drop(columns=[strTarget]), dctConfig["data"]["new_data_path"])
    pthAnswers = pthSaveCsv(dfNew[[strTarget]], dctConfig["data"]["new_data_answers_path"])
    return pthNew, pthAnswers


def subMain() -> None:
    """Command line entry point: read the raw file, clean it, save the cleaned copy."""
    # Define the command line options. Both default to None, which means
    # "use the value from config.yaml".
    objParser = argparse.ArgumentParser(description="Load raw data, clean it and save the cleaned copy.")
    subAddConfigArgument(objParser)
    objParser.add_argument("--input", default=None,
                           help="Read this CSV instead of the source in the config (data.source)")
    objParser.add_argument("--output", default=None, help="Output CSV (default: data.processed_path)")
    objArgs = objParser.parse_args()

    dctConfig = dctLoadConfig(objArgs.config)

    # A command line value wins if one was given. Otherwise use the config.
    strOutputPath = objArgs.output or dctConfig["data"]["processed_path"]

    # Everything printed inside this block goes to the RTF file, not the screen.
    with objCaptureToRtf(dctConfig["outputs"]["rtf_path"], "Ingest"):
        # Do the work, then print a one line summary so a run leaves a visible trace.
        if objArgs.input:
            # An explicit --input is always a local file, whatever data.source says.
            dfData = dfLoadCsv(objArgs.input)
            dctLoad = {"source": "file (--input)", "location": str(objArgs.input),
                       "local_file": str(objArgs.input), "fingerprint": strFileFingerprint(objArgs.input),
                       "loaded_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        else:
            dfData, dctLoad = tplLoadRawData(dctConfig["data"])

        # Display the file header at the top of the stage output
        subPrintPiplineHeader(
            "Ingest",
            dctConfig["project"]["client_question"],
            dctConfig["features"]["target_column"],
            ", ".join(dctConfig["features"]["feature_columns"]),
            dctConfig["data"]["raw_path"],
        )

        # Evidence of where this run's data came from, printed above the profile and the answers.
        print(f"\n{STR_BOLD}{STR_UNDERLINE}Data source{STR_RESET}")
        print(f"{STR_BOLD}Source type:{STR_RESET} {dctLoad['source']}")
        print(f"{STR_BOLD}Location:{STR_RESET} {dctLoad['location']}")
        print(f"{STR_BOLD}Local copy read:{STR_RESET} {dctLoad['local_file']}")
        print(f"{STR_BOLD}Loaded at:{STR_RESET} {dctLoad['loaded_at']}")
        print(f"{STR_BOLD}File fingerprint (SHA-256, first 12):{STR_RESET} {dctLoad['fingerprint']}")

        print(f"\n{STR_BOLD}{STR_UNDERLINE}RAW DATA{STR_RESET}")
        bDisplayDatasetDetails(dfData)

        # The profile above is the evidence for the raw data and where it came from.
        subPrintStep(
            f"\n{STR_BOLD}Step 1 - {STR_RESET}{STR_ITALIC}Pipeline architecture diagram: describe the raw data and its source.{STR_RESET}",
            "The raw data is stored in one CSV file called synthetic_clinical_dataset.csv, which is defined in the config in data.raw_path. The data includes 10,000 rows (one per patient) and 14 columns in total. Six columns are integers, six are decimals, and two are text (sex and diagnosis). The data profile shows no missing values and no duplicate rows. The input variables for the tool are age, sex, BMI, systolic and diastolic blood pressure, glucose, cholesterol, creatinine, diabetes and hypertension. Diagnosis is a text column that stays in the cleaned file but is not used as an input. The target variable is readmission_30d, indicating an average of about 14% of patients were readmitted within 30 days. Mortality is a competing endpoint and would leak information if used as an input variable. The patient_id variable is a label and thus not used as an input variable. The data profile also reveals a problem that will surface later. Age has a minimum value of 0, meaning some records must be for newborns or children. The tool is designed to work with adult inpatients and thus the ingest filters on age.",
        )

        # dfLoadCsv above loads the raw data from the path in config.yaml, the same way every run.
        subPrintStep(
            f"\n{STR_BOLD}Step 2 - {STR_RESET}{STR_ITALIC}Implement a repeatable process for loading the raw data from a file, URL or other defined source.{STR_RESET}",
            "One function, tplLoadRawData, loads the raw data and reads its settings from the config, so every run loads the data the same way.. data.source determines the data source (i.e., whether it is a file or a web address). For files, the raw data is read from data.raw_path. For web addresses, the raw data is downloaded from data.raw_url to the specified path. The download is saved with a temporary file name. Only when the download has completed is the temporary file name replaced with the final file name. Should the download fail for some reason or the resulting file be empty, the run is terminated with a message indicating the problem with the web address or file specified for the test case. In the run report, each run is described by the data source, the location from which the data was loaded, the time it was loaded, and a short file fingerprint. In this case, a local file was read as specified in the config.",
        )

        # dfLoadCsv above reads one file at a fixed path. Say what happens if that file is late, missing or empty.
        subPrintStep(
            f"\n{STR_BOLD}Step 6 - {STR_RESET}{STR_ITALIC}Identify operational risks, such as unavailable or delayed data.{STR_RESET}",
            "The biggest risk is that the raw data for a given file is missing or late. For files the system tries to download, if the file or associated web address is unavailable, or if the file is empty, the system will retry the download a number of times, then stop and provide a clear message. Late data is a gap that currently is not addressed. While load time and a file fingerprint are recorded after ingestion, nothing currently checks how old the file is; as a result, a file could be stale and used without warning. A changed layout is a third risk. If the dataset column changes, the Ingest stops when it can’t find a column listed in the config.",
        )

        # The raw data above is where data authority starts. Say who owns this data and whose consultation would be needed.
        subPrintStep(
            f"\n{STR_BOLD}Step 6 - {STR_RESET}{STR_ITALIC}Consider cultural safety, data authority and the need for consultation where the data, prediction or proposed use could affect Aboriginal and Torres Strait Islander peoples, communities, Country or knowledge.{STR_RESET}",
            "The dataset lacks a column (or combination of colums) that allows identification of Indigenous people. Therefore, it is not possible to evaluate how the model is treating Aboriginal and Torres Strait Islander patients. The dataset would need to include data so performance can be reported for First Nations patients, and any use would need to demonstrate a collective benefit for that community/communities. Regarding authority to control use of the data, this would sit with the relevant Aboriginal community and their health service (e.g., an Aboriginal Community Controlled Health Organisation), as well as the hospital’s data governance and an ethics committee with appropriate Aboriginal representation.",
        )

        # The dataset is synthetic and has no ethnicity or Indigenous status column. Say what a real setting would add.
        subPrintStep(
            f"\n{STR_BOLD}Step 6 - {STR_RESET}{STR_ITALIC}Where cultural safety issues are not directly evident in the selected dataset, explain what would still need to be considered before applying the system in a real organisational or community setting.{STR_RESET}",
            "Although there are no issues with cultural safety identified in this data, a real hospital would record a patient’s self-identified Indigenous status, enabling the model to be checked for Aboriginal and Torres Strait Islander patients. The organisation would also need to consult with and obtain approval from the local Aboriginal community and their health service on how the data will be collected, stored, and used. Patients and staff must be informed that the model exists, how individuals will be flagged, and who can overturn a patient's flagging, with follow-up actions that are culturally safe (e.g. involving Aboriginal health workers). I would also like to set up a review after the system goes live to identify whether any groups have been under-flagged by the model, and to inform the community and organisation so they can take appropriate action where required.",
        )

        # Keep only the columns listed in config.yaml, then clean what is left.
        intRawRows = len(dfData)
        dfData = dfKeepColumns(dfData, dctConfig["data"]["keep_columns"])
        dfData = dfCleanData(dfData)
        intAfterClean = len(dfData)

        # Remove rows with values that are not on the allowed lists (for example, sex = Other).
        dfData = dfFilterAllowedValues(dfData, dctConfig["data"]["allowed_values"] or {})
        intAfterValues = len(dfData)

        # Remove rows outside the value range set in the config (age 18 to 100 as shipped).
        dfData = dfFilterValueRange(dfData, dctConfig["data"]["value_column"],
                                  dctConfig["data"]["min_value"], dctConfig["data"]["max_value"])
        intCleanRows = len(dfData)

        # Set aside some patients as "new" patients, before anything else sees the data.
        dfData, dfNew = tplHoldOutNewPatients(dfData, dctConfig)
        pthNewFile, pthAnswersFile = tplSaveNewPatients(dfNew, dctConfig)
        print(f"\n{STR_BOLD}{STR_UNDERLINE}CLEANED DATA{STR_RESET}")
        bDisplayDatasetDetails(dfData)

        pthOut = pthSaveCsv(dfData, strOutputPath)
        print(f"\nRead {intRawRows} rows. \nRemoved {intRawRows - intAfterClean} duplicates, "
              f"\n{intAfterClean - intAfterValues} rows with other values "
              f"\nand {intAfterValues - intCleanRows} rows outside the {dctConfig['data']['value_column']} range "
              f"({dctConfig['data']['min_value'] if dctConfig['data']['min_value'] is not None else 'no limit'} to "
              f"{dctConfig['data']['max_value'] if dctConfig['data']['max_value'] is not None else 'no limit'}). "
              f"\nSaved {len(dfData)} rows to {pthOut}")
        if pthNewFile is not None:
            print(f"Set aside {len(dfNew)} of {intCleanRows} patients ({dctConfig['data']['new_data_fraction']:.0%}) "
                  f"as new patients. Questions: {pthNewFile}. Answers: {pthAnswersFile}")
        else:
            print("No patients were set aside as new patients (data.new_data_fraction is 0).")

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
             f"maximum value ({dctConfig['data']['max_value']})\n")

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


# This block only runs when the file is executed directly (python -m src.ingest).
# It does not run when another module imports dfLoadCsv or pthSaveCsv.
if __name__ == "__main__":
    subMain()