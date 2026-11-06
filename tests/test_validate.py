"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    test_validate.py

    Tests for src/validate.py. Each test covers one rule: good data passes, and each
    kind of bad data is rejected with the right error.
"""

import pandas as pd
import pytest

from src.validate import ValidationError, lstCheckData, lstRequiredColumns, dfValidate


def test_required_columns_come_from_config(dctConfig):
    """The required columns are the config's feature columns plus its target."""
    assert lstRequiredColumns(dctConfig) == ["feature_a", "feature_b", "target"]


def test_valid_data_passes(dfSample, dctConfig):
    """Good data produces no problems, and dfValidate() hands back the same object."""
    assert lstCheckData(dfSample, dctConfig) == []
    assert dfValidate(dfSample, dctConfig) is dfSample


def test_empty_dataframe_fails(dctConfig):
    """A table with no rows or columns must be rejected."""
    with pytest.raises(ValidationError):
        dfValidate(pd.DataFrame(), dctConfig)


def test_missing_column_fails(dfSample, dctConfig):
    """Dropping a required column must be rejected, and the message must say why."""
    with pytest.raises(ValidationError, match="Missing columns"):
        dfValidate(dfSample.drop(columns=["feature_b"]), dctConfig)


def test_null_values_fail(dfSample, dctConfig):
    """A missing value in a required column must be rejected, naming that column."""
    # Blank out one cell in feature_a.
    dfSample.loc[0, "feature_a"] = None
    with pytest.raises(ValidationError, match="feature_a"):
        dfValidate(dfSample, dctConfig)


def test_too_few_rows_fails(dfSample, dctConfig):
    """Fewer rows than validation.min_rows must be rejected."""
    with pytest.raises(ValidationError, match="at least 10"):
        dfValidate(dfSample.head(5), dctConfig)


def test_min_rows_setting_changes_the_result(dfSample, dctConfig):
    """Raising min_rows in the config makes the same data fail.

    This shows the rule is driven by the config and not fixed in the code.
    """
    dctConfig["validation"]["min_rows"] = 50
    with pytest.raises(ValidationError, match="at least 50"):
        dfValidate(dfSample, dctConfig)


def test_the_range_check_is_skipped_when_no_range_is_set(dctConfig):
    """With min_value and max_value both null, a missing value_column is not a problem."""
    import pandas as pd
    from src.validate import lstCheckData
    dctConfig["data"]["min_value"] = None
    dctConfig["data"]["max_value"] = None
    dctConfig["data"]["value_column"] = "age"
    dfData = pd.DataFrame({"feature_a": range(20), "feature_b": range(20), "target": [0, 1] * 10})
    assert lstCheckData(dfData, dctConfig) == []


# ---------------------------------------------------------------------------
# Checks 9 to 11: unexpected columns, numeric types and column ranges
# ---------------------------------------------------------------------------
def test_an_unexpected_column_fails(dfSample, dctConfig):
    """A column that data.keep_columns doesn't list is reported."""
    dfSample["surprise"] = 1
    lstProblems = lstCheckData(dfSample, dctConfig)
    assert any("Unexpected columns" in strProblem and "surprise" in strProblem for strProblem in lstProblems)


def test_a_text_value_in_a_numeric_column_fails(dfSample, dctConfig):
    """A numeric feature column that holds text is reported."""
    dfSample["feature_a"] = dfSample["feature_a"].astype(str)
    assert any("Not numeric" in strProblem for strProblem in lstCheckData(dfSample, dctConfig))


def test_values_outside_a_column_range_fail(dfSample, dctConfig):
    """Values below min or above max in column_ranges are counted."""
    dctConfig["validation"]["column_ranges"] = {"feature_b": {"min": 5, "max": 10}}
    dfSample["feature_b"] = list(range(len(dfSample)))
    lstProblems = lstCheckData(dfSample, dctConfig)
    assert any("feature_b has" in strProblem for strProblem in lstProblems)


def test_values_inside_a_column_range_pass(dfSample, dctConfig):
    """Data inside the limits gives no range problem."""
    dctConfig["validation"]["column_ranges"] = {"feature_b": {"min": -1000, "max": 1000}}
    assert lstCheckData(dfSample, dctConfig) == []


def test_an_open_ended_range_only_checks_one_side(dfSample, dctConfig):
    """With only a max, small values are fine and large ones fail."""
    dctConfig["validation"]["column_ranges"] = {"feature_b": {"max": 0}}
    dfSample["feature_b"] = 5
    assert any("feature_b has 0 values below None and" in strProblem for strProblem in lstCheckData(dfSample, dctConfig))


def test_a_range_for_a_missing_column_fails(dfSample, dctConfig):
    """A limit for a column that isn't in the data is a problem, not a silent pass."""
    dctConfig["validation"]["column_ranges"] = {"nope": {"min": 0}}
    assert any("nope is not in the data" in strProblem for strProblem in lstCheckData(dfSample, dctConfig))
