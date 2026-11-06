"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    test_features.py
    
    Tests for src/features.py: building dfX and serY, and splitting into train and test.
"""

import pandas as pd
import pytest

from src.features import dfEngineerFeatures, tplBuildFeatures, lstSplitData


def test_build_features_returns_x_and_y(dfSample, dctConfig):
    """dfX holds only the feature columns, and dfX and serY have the same number of rows."""
    dfX, serY = tplBuildFeatures(dfSample, dctConfig)
    assert list(dfX.columns) == dctConfig["features"]["feature_columns"]
    assert len(dfX) == len(serY) == len(dfSample)


def test_feature_columns_follow_the_config(dfSample, dctConfig):
    """Changing the column list in the config changes what dfX contains."""
    dctConfig["features"]["feature_columns"] = ["feature_a"]
    dfX, _ = tplBuildFeatures(dfSample, dctConfig)
    assert list(dfX.columns) == ["feature_a"]


def test_split_sizes(dfSample, dctConfig):
    """A 25% test split of 20 rows gives 5 test rows and 15 training rows."""
    dctConfig["features"]["test_size"] = 0.25
    dfX, serY = tplBuildFeatures(dfSample, dctConfig)
    dfXTrain, dfXTest, serYTrain, serYTest = lstSplitData(dfX, serY, dctConfig)
    assert len(dfXTest) == 5
    assert len(dfXTrain) == 15
    # The target must be split in step with the features.
    assert len(serYTrain) == 15
    assert len(serYTest) == 5


def test_split_is_repeatable(dfSample, dctConfig):
    """Two splits of the same data pick the same test rows.

    This matters because train.py and evaluate.py split separately. If the split
    changed between runs, evaluation could score the model on training rows.
    """
    dfX, serY = tplBuildFeatures(dfSample, dctConfig)
    dfFirst = lstSplitData(dfX, serY, dctConfig)[1]   # Index 1 is dfXTest.
    dfSecond = lstSplitData(dfX, serY, dctConfig)[1]
    assert list(dfFirst.index) == list(dfSecond.index)


def test_stratified_split_keeps_the_target_mix_in_both_sets(dctConfig):
    """With stratify on, the train and test rows have the same share of 1s as the whole data."""
    # 100 rows, 10% are 1s. A plain random split could easily give 7% or 14%.
    dfData = pd.DataFrame({"feature_a": range(100), "feature_b": 1, "target": [1] * 10 + [0] * 90})
    dfX, serY = tplBuildFeatures(dfData, dctConfig)
    _, _, serYTrain, serYTest = lstSplitData(dfX, serY, dctConfig)
    assert serYTrain.mean() == 0.1
    assert serYTest.mean() == 0.1


def test_stratify_can_be_switched_off_in_the_config(dctConfig):
    """With stratify off, the split still works and still has the requested sizes."""
    dctConfig["features"]["stratify"] = False
    dfData = pd.DataFrame({"feature_a": range(100), "feature_b": 1, "target": [1] * 10 + [0] * 90})
    dfX, serY = tplBuildFeatures(dfData, dctConfig)
    dfXTrain, dfXTest, _, _ = lstSplitData(dfX, serY, dctConfig)
    assert len(dfXTest) == 20 and len(dfXTrain) == 80


def dctPatientConfig(dctConfig: dict) -> dict:
    """Test helper: set the rules for pulse pressure and the sex encoding."""
    dctConfig["features"]["difference_columns"] = {"pulse_pressure": ["sys", "dia"]}
    dctConfig["features"]["categorical_columns"] = {"sex": ["Female", "Male"]}
    return dctConfig


def test_pulse_pressure_is_systolic_minus_diastolic(dctConfig):
    """The difference rule creates the new column with the right values."""
    dfIn = pd.DataFrame({"sys": [120, 140], "dia": [80, 90], "sex": ["Male", "Female"]})
    dfOut = dfEngineerFeatures(dfIn, dctPatientConfig(dctConfig))
    assert list(dfOut["pulse_pressure"]) == [40, 50]


def test_sex_becomes_one_zero_column_and_the_text_column_goes(dctConfig):
    """Female is the reference (0) and Male is 1, and the original text column is dropped."""
    dfIn = pd.DataFrame({"sys": [120, 140], "dia": [80, 90], "sex": ["Male", "Female"]})
    dfOut = dfEngineerFeatures(dfIn, dctPatientConfig(dctConfig))
    assert "sex" not in dfOut.columns
    assert list(dfOut["sex_Male"]) == [1, 0]


def test_one_category_batch_gets_the_same_columns(dctConfig):
    """A batch with only women still gets sex_Male (all 0), so columns match training."""
    dfBoth = pd.DataFrame({"sys": [120, 140], "dia": [80, 90], "sex": ["Male", "Female"]})
    dfOne = pd.DataFrame({"sys": [130], "dia": [85], "sex": ["Female"]})
    dctC = dctPatientConfig(dctConfig)
    assert list(dfEngineerFeatures(dfOne, dctC).columns) == list(dfEngineerFeatures(dfBoth, dctC).columns)
    assert list(dfEngineerFeatures(dfOne, dctC)["sex_Male"]) == [0]


def test_unknown_category_raises(dctConfig):
    """A value not on the configured list stops the run instead of silently becoming 0."""
    dfIn = pd.DataFrame({"sys": [120], "dia": [80], "sex": ["Other"]})
    with pytest.raises(ValueError, match="sex"):
        dfEngineerFeatures(dfIn, dctPatientConfig(dctConfig))


def test_blank_category_raises(dctConfig):
    """A missing sex value also stops the run."""
    dfIn = pd.DataFrame({"sys": [120], "dia": [80], "sex": [None]})
    with pytest.raises(ValueError, match="sex"):
        dfEngineerFeatures(dfIn, dctPatientConfig(dctConfig))


def test_engineering_does_not_change_the_input(dctConfig):
    """The function works on a copy, so the caller's data stays as it was."""
    dfIn = pd.DataFrame({"sys": [120], "dia": [80], "sex": ["Male"]})
    dfEngineerFeatures(dfIn, dctPatientConfig(dctConfig))
    assert list(dfIn.columns) == ["sys", "dia", "sex"]
