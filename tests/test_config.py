"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    test_config.py
    
    Tests for src/config.py: loading config.yaml and rejecting a bad one.
"""

from pathlib import Path

import pytest
import yaml

from src.config import (
    STR_CONFIG_ENV_VAR,
    DCT_REQUIRED_KEYS,
    ConfigError,
    dctLoadConfig,
    subCheckConfig,
)

# The real config.yaml at the project root. Built from this file's location so
# the test works no matter which folder pytest is run from.
PTH_REAL_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def pthWriteYaml(pthFile: Path, dctData) -> Path:
    """Test helper: write a dictionary to a YAML file and return its path."""
    pthFile.write_text(yaml.safe_dump(dctData))
    return pthFile


def test_real_config_loads_with_every_required_key():
    """The config.yaml shipped with the project has every section and key."""
    dctLoaded = dctLoadConfig(PTH_REAL_CONFIG)
    for strSection, lstKeys in DCT_REQUIRED_KEYS.items():
        for strKey in lstKeys:
            assert strKey in dctLoaded[strSection]


def test_values_are_read_from_the_file(dctConfig, tmp_path):
    """The loader returns the values that are in the file, unchanged."""
    dctConfig["data"]["raw_path"] = "somewhere/else.csv"
    pthFile = pthWriteYaml(tmp_path / "custom.yaml", dctConfig)
    assert dctLoadConfig(pthFile)["data"]["raw_path"] == "somewhere/else.csv"


def test_missing_file_raises(tmp_path):
    """A config path that doesn't exist gives a clear FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        dctLoadConfig(tmp_path / "nope.yaml")


def test_missing_key_raises(dctConfig, tmp_path):
    """Removing a required key is rejected, and the message names the key."""
    del dctConfig["model"]["max_iter"]
    pthFile = pthWriteYaml(tmp_path / "bad.yaml", dctConfig)
    with pytest.raises(ConfigError, match="model.max_iter"):
        dctLoadConfig(pthFile)


def test_missing_section_raises(dctConfig, tmp_path):
    """Removing a whole section is rejected, and the message names the section."""
    del dctConfig["outputs"]
    pthFile = pthWriteYaml(tmp_path / "bad.yaml", dctConfig)
    with pytest.raises(ConfigError, match="outputs"):
        dctLoadConfig(pthFile)


def test_empty_file_raises(tmp_path):
    """An empty config file is rejected rather than loading as None."""
    pthFile = tmp_path / "empty.yaml"
    pthFile.write_text("")
    with pytest.raises(ConfigError):
        dctLoadConfig(pthFile)


def test_empty_feature_list_raises(dctConfig, tmp_path):
    """A model with no input columns can't work, so the loader rejects it."""
    dctConfig["features"]["feature_columns"] = []
    pthFile = pthWriteYaml(tmp_path / "bad.yaml", dctConfig)
    with pytest.raises(ConfigError, match="feature_columns"):
        dctLoadConfig(pthFile)


def test_environment_variable_selects_the_file(dctConfig, tmp_path, monkeypatch):
    """With no path given, the PIPELINE_CONFIG variable decides which file loads.

    monkeypatch sets the variable for this one test only and then puts things
    back, so it cannot leak into other tests.
    """
    dctConfig["data"]["raw_path"] = "from_env.csv"
    pthFile = pthWriteYaml(tmp_path / "env.yaml", dctConfig)
    monkeypatch.setenv(STR_CONFIG_ENV_VAR, str(pthFile))
    assert dctLoadConfig()["data"]["raw_path"] == "from_env.csv"


def test_real_config_has_separate_rtf_paths_for_runs_and_tests():
    """Tests save their RTF files somewhere else, so a test run never overwrites a real run's files."""
    dctLoaded = dctLoadConfig(PTH_REAL_CONFIG)
    assert dctLoaded["outputs"]["rtf_path"] != dctLoaded["testing"]["rtf_path"]
    assert "{stage}" in dctLoaded["outputs"]["rtf_path"]
    assert "{stage}" in dctLoaded["testing"]["rtf_path"]


def test_missing_rtf_path_raises(dctConfig, tmp_path):
    """Removing outputs.rtf_path is rejected, and the message names the key."""
    del dctConfig["outputs"]["rtf_path"]
    pthFile = pthWriteYaml(tmp_path / "bad.yaml", dctConfig)
    with pytest.raises(ConfigError, match="outputs.rtf_path"):
        dctLoadConfig(pthFile)


def test_real_config_lists_the_new_model_and_evaluation_settings():
    """config.yaml has class_weight, positive_label, and sex is encoded by a rule."""
    dctLoaded = dctLoadConfig(PTH_REAL_CONFIG)
    assert "class_weight" in dctLoaded["model"]
    assert dctLoaded["evaluation"]["positive_label"] == 1
    assert "sex" in dctLoaded["features"]["categorical_columns"]


def test_excluded_column_in_feature_list_raises(dctConfig, tmp_path):
    """Listing a banned column (such as mortality) as a feature stops the run."""
    dctConfig["features"]["excluded_columns"] = ["feature_a"]
    pthFile = pthWriteYaml(tmp_path / "c.yaml", dctConfig)
    with pytest.raises(ConfigError, match="excluded"):
        dctLoadConfig(pthFile)


def test_difference_rule_with_unlisted_column_raises(dctConfig, tmp_path):
    """A difference rule that uses a column missing from feature_columns stops the run."""
    dctConfig["features"]["difference_columns"] = {"gap": ["feature_a", "nope"]}
    pthFile = pthWriteYaml(tmp_path / "c.yaml", dctConfig)
    with pytest.raises(ConfigError, match="feature rule"):
        dctLoadConfig(pthFile)


def test_categorical_rule_with_unlisted_column_raises(dctConfig, tmp_path):
    """A categorical rule for a column missing from feature_columns stops the run."""
    dctConfig["features"]["categorical_columns"] = {"nope": ["a", "b"]}
    pthFile = pthWriteYaml(tmp_path / "c.yaml", dctConfig)
    with pytest.raises(ConfigError, match="feature rule"):
        dctLoadConfig(pthFile)


@pytest.mark.parametrize("varBad", [-0.1, 0.5, 0.9, "five", True])
def test_bad_new_data_fraction_raises(dctConfig, tmp_path, varBad):
    """The set-aside share must be a number from 0 up to, but not including, 0.5."""
    dctConfig["data"]["new_data_fraction"] = varBad
    pthFile = pthWriteYaml(tmp_path / "c.yaml", dctConfig)
    with pytest.raises(ConfigError, match="new_data_fraction"):
        dctLoadConfig(pthFile)


@pytest.mark.parametrize("varBad", ["eighteen", True])
def test_bad_min_value_raises(dctConfig, tmp_path, varBad):
    """The lower limit must be a number or null."""
    dctConfig["data"]["min_value"] = varBad
    pthFile = pthWriteYaml(tmp_path / "c.yaml", dctConfig)
    with pytest.raises(ConfigError, match="min_value"):
        dctLoadConfig(pthFile)


@pytest.mark.parametrize("varBad", ["hundred", True])
def test_bad_max_value_raises(dctConfig, tmp_path, varBad):
    """The upper limit must be a number or null."""
    dctConfig["data"]["max_value"] = varBad
    pthFile = pthWriteYaml(tmp_path / "c.yaml", dctConfig)
    with pytest.raises(ConfigError, match="max_value"):
        dctLoadConfig(pthFile)


def test_max_value_below_min_value_raises(dctConfig, tmp_path):
    """An upper limit under the lower limit would remove everyone."""
    dctConfig["data"]["min_value"] = 40
    dctConfig["data"]["max_value"] = 30
    dctConfig["data"]["keep_columns"] = ["age"] + dctConfig["data"]["keep_columns"]
    pthFile = pthWriteYaml(tmp_path / "c.yaml", dctConfig)
    with pytest.raises(ConfigError, match="can't be below"):
        dctLoadConfig(pthFile)


@pytest.mark.parametrize("strEnd", ["min_value", "max_value"])
def test_age_limit_needs_the_value_column_to_be_kept(dctConfig, tmp_path, strEnd):
    """Setting either end of the range needs the age column in keep_columns."""
    dctConfig["data"][strEnd] = 50
    pthFile = pthWriteYaml(tmp_path / "c.yaml", dctConfig)
    with pytest.raises(ConfigError, match="value_column"):
        dctLoadConfig(pthFile)


# ---------------------------------------------------------------------------
# data.source, raw_url and the download settings
# ---------------------------------------------------------------------------
def test_source_must_be_file_or_url(dctConfig, tmp_path):
    """Any other source name is refused."""
    dctConfig["data"]["source"] = "ftp"
    pthFile = tmp_path / "c.yaml"
    pthFile.write_text(yaml.safe_dump(dctConfig))
    with pytest.raises(ConfigError, match="data.source"):
        dctLoadConfig(pthFile)


def test_a_url_source_needs_an_http_address(dctConfig, tmp_path):
    """source url with no address, or a non-http address, is refused."""
    dctConfig["data"]["source"] = "url"
    dctConfig["data"]["raw_url"] = None
    pthFile = tmp_path / "c.yaml"
    pthFile.write_text(yaml.safe_dump(dctConfig))
    with pytest.raises(ConfigError, match="raw_url"):
        dctLoadConfig(pthFile)
    dctConfig["data"]["raw_url"] = "file:///x.csv"
    pthFile.write_text(yaml.safe_dump(dctConfig))
    with pytest.raises(ConfigError, match="raw_url"):
        dctLoadConfig(pthFile)


def test_a_valid_url_source_loads(dctConfig, tmp_path):
    """source url with an https address passes the check."""
    dctConfig["data"]["source"] = "url"
    dctConfig["data"]["raw_url"] = "https://example.org/d.csv"
    pthFile = tmp_path / "c.yaml"
    pthFile.write_text(yaml.safe_dump(dctConfig))
    assert dctLoadConfig(pthFile)["data"]["source"] == "url"


def test_download_settings_must_be_whole_numbers_of_at_least_one(dctConfig, tmp_path):
    """Zero, negative, text and True are refused for the timeout and retries."""
    pthFile = tmp_path / "c.yaml"
    for strKey in ("download_timeout", "download_retries"):
        for varBad in (0, -1, "ten", True):
            dctBad = {**dctConfig, "data": {**dctConfig["data"], strKey: varBad}}
            pthFile.write_text(yaml.safe_dump(dctBad))
            with pytest.raises(ConfigError, match=strKey):
                dctLoadConfig(pthFile)


def test_column_ranges_must_have_numbers_and_max_not_below_min(dctConfig, tmp_path):
    """Bad limits in validation.column_ranges are refused."""
    pthFile = tmp_path / "c.yaml"
    for dctBad in ({"bmi": {"min": 10, "max": 5}}, {"bmi": {"min": "low"}}, {"bmi": {"top": 5}}, {"bmi": 5}):
        dctNew = {**dctConfig, "validation": {**dctConfig["validation"], "column_ranges": dctBad}}
        pthFile.write_text(yaml.safe_dump(dctNew))
        with pytest.raises(ConfigError, match="column_ranges"):
            dctLoadConfig(pthFile)


def test_valid_column_ranges_load(dctConfig, tmp_path):
    """Limits with one open end load fine."""
    dctConfig["validation"]["column_ranges"] = {"bmi": {"min": 10}, "age": {"max": 100}}
    pthFile = tmp_path / "c.yaml"
    pthFile.write_text(yaml.safe_dump(dctConfig))
    assert "bmi" in dctLoadConfig(pthFile)["validation"]["column_ranges"]


def test_form_defaults_must_name_feature_columns(dctConfig):
    """A form default for a column that isn't a feature is refused."""
    dctConfig["service"]["form_defaults"] = {"not_a_column": 1}
    with pytest.raises(ConfigError, match="form_defaults"):
        subCheckConfig(dctConfig)
