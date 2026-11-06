"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    test_compare.py

    Tests for src/compare.py: the baseline and candidate comparison table.
"""

import numpy as np
import pandas as pd
import pytest

from src.compare import dfCompareModels
from src.features import tplBuildFeatures, lstSplitData


def dfMakeData(intRows: int = 200) -> pd.DataFrame:
    """Test helper: a data set where feature_a carries some signal and about 20 percent are positive."""
    objRng = np.random.default_rng(0)
    serTarget = pd.Series((objRng.random(intRows) < 0.2).astype(int))
    return pd.DataFrame({
        "feature_a": objRng.normal(size=intRows) + serTarget * 1.5,
        "feature_b": objRng.normal(size=intRows),
        "target": serTarget,
    })


def dfRun(dctConfig: dict) -> pd.DataFrame:
    """Test helper: build features, split, and compare."""
    dfX, serY = tplBuildFeatures(dfMakeData(), dctConfig)
    dfXTrain, dfXTest, serYTrain, serYTest = lstSplitData(dfX, serY, dctConfig)
    return dfCompareModels(dfXTrain, dfXTest, serYTrain, serYTest, dctConfig), serYTest


def test_one_row_per_model_with_baseline_first(dctConfig):
    """The table has the baseline, then the rule benchmark, then each candidate in config order."""
    dfResults, _ = dfRun(dctConfig)
    assert list(dfResults["model"]) == ["baseline", "rule_a", "logistic_regression", "random_forest"]
    assert list(dfResults["role"]) == ["baseline", "benchmark", "candidate", "candidate"]


def test_baseline_matches_the_majority_share_and_finds_no_positives(dctConfig):
    """Always predicting the commoner outcome gives accuracy equal to its share, with zero recall."""
    dfResults, serYTest = dfRun(dctConfig)
    dfBase = dfResults[dfResults["model"] == "baseline"].iloc[0]
    assert dfBase["accuracy"] == pytest.approx(max(serYTest.mean(), 1 - serYTest.mean()))
    assert dfBase["recall"] == 0
    assert dfBase["roc_auc"] == pytest.approx(0.5)


def test_a_model_with_signal_beats_the_baseline_on_auc(dctConfig):
    """The logistic candidate finds the signal in feature_a, so its AUC is above 0.5."""
    dfResults, _ = dfRun(dctConfig)
    assert dfResults.set_index("model").loc["logistic_regression", "roc_auc"] > 0.6


def test_cross_validation_columns_are_present(dctConfig):
    """Each row carries the cross-validated AUC mean and spread."""
    dfResults, _ = dfRun(dctConfig)
    assert {"cv_auc_mean", "cv_auc_std"} <= set(dfResults.columns)


def test_unknown_candidate_type_raises(dctConfig):
    """A candidate with a type the code doesn't know stops the run."""
    dctConfig["comparison"]["candidates"]["bad"] = {"type": "magic"}
    with pytest.raises(ValueError):
        dfRun(dctConfig)
