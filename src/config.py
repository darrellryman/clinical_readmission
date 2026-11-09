"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    config.py

    Loads and checks config.yaml, the single place where the pipeline's settings live.

    Every other script calls dctLoadConfig() and reads its values from the dictionary
    it returns, so paths, column names and model settings are never hard-coded in
    the pipeline logic.

    Which file gets loaded, in order of priority:
        1. The path passed in (for the scripts, the --config option).
        2. The PIPELINE_CONFIG environment variable.
        3. config.yaml in the current folder (the project root).

    Naming convention (Hungarian notation), used in every file in src/:
        Variables start with a short prefix for their type, then the name in
        CamelCase, for example strInputPath or dfData.
            str  text              int  whole number     flt  decimal number
            bln  True or False     lst  list             dct  dictionary
            tpl  tuple             df   pandas DataFrame ser  pandas Series
            pth  pathlib Path      mdl  fitted model     obj  any other object
            var  variant: more than one type is accepted (for example str or Path)
        Constants are written in capitals with the type prefix first, for example
        PTH_DEFAULT_CONFIG.
        Procedures start with the type they return, then the action, for example
        dfLoadCsv() returns a DataFrame. A procedure that returns nothing starts
        with "sub", for example subMain().
        Class names (ConfigError) and the names of parameters that scikit-learn,
        argparse and pandas define themselves are left as they are.
"""

import os
from pathlib import Path

# PyYAML parses the config file into ordinary Python dictionaries and lists.
import yaml

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# The file used when nothing else is specified.
PTH_DEFAULT_CONFIG = Path("config.yaml")

# The environment variable that can point to a different config file.
STR_CONFIG_ENV_VAR = "PIPELINE_CONFIG"

# Every section and key the pipeline needs. If any are missing from the file,
# dctLoadConfig() stops with one message listing all of them, instead of the
# pipeline failing later with an unhelpful KeyError.
DCT_REQUIRED_KEYS = {
    "project": ["random_state", "client_question"],
    "data": [
        "source", "raw_path", "raw_url", "download_timeout", "download_retries",
        "processed_path", "keep_columns", "allowed_values",
        "new_data_fraction", "new_data_path", "new_data_answers_path",
        "min_value", "max_value", "value_column",
    ],
    "validation": ["min_rows", "column_ranges"],
    "features": [
        "feature_columns", "target_column", "test_size", "stratify",
        "categorical_columns", "difference_columns", "excluded_columns",
    ],
    "model": ["path", "max_iter", "C", "class_weight"],
    "outputs": ["metrics_path", "predictions_path", "prediction_column", "rtf_path", "comparison_path", "features_path"],
    "evaluation": ["average", "positive_label", "group_columns", "binned_columns"],
    "comparison": ["cv_folds", "baseline", "benchmarks", "candidates"],
    "approval": [
        "enabled", "min_auc_gain", "model_card_path", "evidence_path",
        "intended_use", "not_for", "known_limits", "recommendation",
    ],
    "plots": ["enabled", "folder", "dpi", "palette", "group_metrics"],
    "service": ["host", "port", "max_patients"],
    "tracking": ["enabled", "uri", "experiment_name"],
    "monitoring": [
        "psi_watch", "psi_alert", "psi_bins", "bad_share_watch", "bad_share_alert",
        "flag_rate_watch", "flag_rate_alert", "min_batch_rows",
    ],
    "testing": ["rtf_path"],
}


class ConfigError(Exception):
    """Raised when the config file is present but its contents are not usable."""


def subAddConfigArgument(objParser) -> None:
    """Add the shared --config option to a script's argument parser.

    Every script calls this, so they all accept --config in the same way.
    """
    objParser.add_argument(
        "--config",
        default=None,
        help="Path to the config file (default: PIPELINE_CONFIG or config.yaml)",
    )


def pthResolveConfigPath(varPath=None) -> Path:
    """Work out which config file to use, following the priority order above."""
    if varPath:
        return Path(varPath)

    # Next, check for an environment variable. The "or None" turns an empty
    # string into None, so an empty variable is treated as not set.
    strEnvValue = os.environ.get(STR_CONFIG_ENV_VAR) or None
    if strEnvValue:
        return Path(strEnvValue)

    return PTH_DEFAULT_CONFIG


def subCheckConfig(dctConfig) -> None:
    """Confirm the loaded config has every section and key the pipeline needs.

    Raises:
        ConfigError: Listing everything that is wrong, all in one message.
    """
    # An empty file loads as None, and a list or a plain value is not a mapping
    # of sections. Either way there is nothing to check further.
    if not isinstance(dctConfig, dict):
        raise ConfigError("Config file is empty or is not a set of named sections")

    lstProblems = []
    for strSection, lstKeys in DCT_REQUIRED_KEYS.items():
        # A section that is missing, or that has no keys under it, can't be checked further.
        if not isinstance(dctConfig.get(strSection), dict):
            lstProblems.append(f"missing section '{strSection}'")
            continue
        for strKey in lstKeys:
            if strKey not in dctConfig[strSection]:
                lstProblems.append(f"missing key '{strSection}.{strKey}'")

    # The model needs at least one input column, so an empty list is an error.
    lstFeatureColumns = dctConfig.get("features", {}).get("feature_columns")
    if "features" in dctConfig and isinstance(lstFeatureColumns, list) and not lstFeatureColumns:
        lstProblems.append("'features.feature_columns' must list at least one column")

    # The feature rules only make sense if the columns they use are real inputs,
    # and a column marked as off limits must never be fed to the model.
    dctFeatures = dctConfig.get("features")
    if isinstance(dctFeatures, dict) and isinstance(lstFeatureColumns, list):
        for strForbidden in dctFeatures.get("excluded_columns") or []:
            if strForbidden in lstFeatureColumns:
                lstProblems.append(
                    f"'{strForbidden}' is in features.excluded_columns, so it can't also be in features.feature_columns"
                )
        lstCalculated = [strCol for lstPair in (dctFeatures.get("difference_columns") or {}).values() for strCol in lstPair]
        lstEncoded = list((dctFeatures.get("categorical_columns") or {}).keys())
        for strCol in lstCalculated + lstEncoded:
            if strCol not in lstFeatureColumns:
                lstProblems.append(f"'{strCol}' is used by a feature rule but is not in features.feature_columns")

    # The share set aside as new patients has to leave plenty of rows for training.
    dctData = dctConfig.get("data")
    if isinstance(dctData, dict) and "new_data_fraction" in dctData:
        varFraction = dctData["new_data_fraction"]
        if isinstance(varFraction, bool) or not isinstance(varFraction, (int, float)) or not 0 <= varFraction < 0.5:
            lstProblems.append("'data.new_data_fraction' must be a number from 0 up to (not including) 0.5")

    # Where the raw data comes from. A url source needs an http or https address,
    # and the timeout and retry count have to be whole numbers of at least 1.
    if isinstance(dctData, dict) and "source" in dctData:
        if dctData["source"] not in ("file", "url", "kaggle"):
            lstProblems.append("'data.source' must be file, url or kaggle")
        elif dctData["source"] == "kaggle":
            for strKey in ("kaggle_dataset", "kaggle_file"):
                if not isinstance(dctData.get(strKey), str) or not dctData[strKey].strip():
                    lstProblems.append(f"'data.{strKey}' must be text when data.source is kaggle")
        elif dctData["source"] == "url":
            varUrl = dctData.get("raw_url")
            if not isinstance(varUrl, str) or not varUrl.lower().startswith(("http://", "https://")):
                lstProblems.append("'data.raw_url' must be an http or https address when data.source is url")
        for strKey in ("download_timeout", "download_retries"):
            varValue = dctData.get(strKey)
            if isinstance(varValue, bool) or not isinstance(varValue, int) or varValue < 1:
                lstProblems.append(f"'data.{strKey}' must be a whole number of at least 1")

    # The per column limits used by validate.py. Each column maps to min and/or max,
    # which are numbers or null, and max can't be below min.
    dctValidation = dctConfig.get("validation")
    if isinstance(dctValidation, dict) and "column_ranges" in dctValidation:
        varRanges = dctValidation["column_ranges"]
        if varRanges is not None and not isinstance(varRanges, dict):
            lstProblems.append("'validation.column_ranges' must be a list of columns with min and max, or {}")
        for strColumn, dctLimits in (varRanges or {}).items():
            if not isinstance(dctLimits, dict) or not set(dctLimits) <= {"min", "max"}:
                lstProblems.append(f"'validation.column_ranges.{strColumn}' must hold only min and max")
                continue
            lstValues = [dctLimits.get("min"), dctLimits.get("max")]
            if any(v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))) for v in lstValues):
                lstProblems.append(f"'validation.column_ranges.{strColumn}' limits must be numbers or null")
            elif None not in lstValues and lstValues[1] < lstValues[0]:
                lstProblems.append(f"'validation.column_ranges.{strColumn}': max can't be below min")

    # The web form's starting values are optional. When given, each key must be a
    # feature column, so a typo can't quietly leave a box blank.
    dctService = dctConfig.get("service")
    if isinstance(dctService, dict) and dctService.get("form_defaults") is not None:
        varDefaults = dctService["form_defaults"]
        if not isinstance(varDefaults, dict):
            lstProblems.append("'service.form_defaults' must be a list of columns with values, or {}")
        elif isinstance(lstFeatureColumns, list):
            for strColumn in varDefaults:
                if strColumn not in lstFeatureColumns:
                    lstProblems.append(f"'service.form_defaults.{strColumn}' is not in features.feature_columns")

    # The value range filter. min_value and max_value are each null (that end is open)
    # or a number, and max_value can't be below min_value.
    if isinstance(dctData, dict) and "min_value" in dctData and "max_value" in dctData:
        varMin, varMax = dctData["min_value"], dctData["max_value"]

        def blnOk(varValue) -> bool:
            """True if the value is null or a real number (not True or False)."""
            return varValue is None or (isinstance(varValue, (int, float)) and not isinstance(varValue, bool))

        if not blnOk(varMin):
            lstProblems.append("'data.min_value' must be a number, or null to leave the low end open")
        if not blnOk(varMax):
            lstProblems.append("'data.max_value' must be a number, or null to leave the high end open")
        if blnOk(varMin) and blnOk(varMax):
            if varMin is not None and varMax is not None and varMax < varMin:
                lstProblems.append("'data.max_value' can't be below 'data.min_value'")
            if (varMin is not None or varMax is not None) \
                    and dctData.get("value_column") not in (dctData.get("keep_columns") or []):
                lstProblems.append("'data.value_column' must be one of data.keep_columns when min_value or max_value is set")

    # Monitoring thresholds. Each watch level must be a number and no higher than its alert level.
    dctMonitoring = dctConfig.get("monitoring")
    if isinstance(dctMonitoring, dict):
        for strWatch, strAlert in (("psi_watch", "psi_alert"), ("bad_share_watch", "bad_share_alert"),
                                   ("flag_rate_watch", "flag_rate_alert")):
            varWatch, varAlert = dctMonitoring.get(strWatch), dctMonitoring.get(strAlert)
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 for v in (varWatch, varAlert)):
                if varWatch is not None and varAlert is not None:
                    lstProblems.append(f"'monitoring.{strWatch}' and 'monitoring.{strAlert}' must be numbers of 0 or more")
            elif varWatch > varAlert:
                lstProblems.append(f"'monitoring.{strWatch}' can't be above 'monitoring.{strAlert}'")
        for strKey in ("psi_bins", "min_batch_rows"):
            varValue = dctMonitoring.get(strKey)
            if varValue is not None and (isinstance(varValue, bool) or not isinstance(varValue, int) or varValue < 1):
                lstProblems.append(f"'monitoring.{strKey}' must be a whole number of at least 1")

    if lstProblems:
        raise ConfigError("; ".join(lstProblems))


def dctLoadConfig(varPath=None) -> dict:
    """Read the config file, check it, and return it as a dictionary.

    Args:
        varPath: Optional path to a config file, as text or a Path. If not
            given, the environment variable and then the default are used.

    Returns:
        The settings, for example dctConfig["data"]["raw_path"].

    Raises:
        FileNotFoundError: If the config file does not exist.
        ConfigError: If the file is missing required sections or keys.
    """
    pthConfig = pthResolveConfigPath(varPath)

    if not pthConfig.exists():
        raise FileNotFoundError(f"Config file not found: {pthConfig}")

    # safe_load only builds plain data (dicts, lists, numbers, text). It will
    # not run code that might be hidden in a YAML file, unlike load().
    with open(pthConfig, "r", encoding="utf-8") as objHandle:
        dctConfig = yaml.safe_load(objHandle)

    subCheckConfig(dctConfig)
    return dctConfig
