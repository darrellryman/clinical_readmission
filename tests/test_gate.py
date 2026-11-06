"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    test_gate.py

    Tests for the approval gate, the model card, the rule benchmark and the
    evaluation evidence record (src/train.py and src/compare.py).
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.compare import dfCompareModels, strBuildEvidenceRecord
from src.features import tplBuildFeatures, lstSplitData
from src.train import (
    RuleBenchmark, dctBuildModelCard, dctCheckModelGate, mdlBuildModel, pthSaveModelCard,
)


def dfMakeData(fltSignal: float, intRows: int = 300) -> pd.DataFrame:
    """Test helper: about 20 percent positive rows, with feature_a carrying fltSignal of signal."""
    objRng = np.random.default_rng(1)
    serTarget = pd.Series((objRng.random(intRows) < 0.2).astype(int))
    return pd.DataFrame({
        "feature_a": objRng.normal(size=intRows) + serTarget * fltSignal,
        "feature_b": objRng.normal(size=intRows),
        "target": serTarget,
    })


def tplTrainRows(dctConfig: dict, fltSignal: float):
    """Test helper: build features and return the training and test parts."""
    dfX, serY = tplBuildFeatures(dfMakeData(fltSignal), dctConfig)
    return lstSplitData(dfX, serY, dctConfig)


def test_gate_passes_when_the_model_beats_the_baseline(dctConfig):
    """A strong signal gives a big AUC gain, so the gate passes."""
    dfXTrain, _, serYTrain, _ = tplTrainRows(dctConfig, 2.0)
    dctGate = dctCheckModelGate(dfXTrain, serYTrain, dctConfig)
    assert dctGate["passed"] is True
    assert dctGate["auc_gain"] > dctConfig["approval"]["min_auc_gain"]


def test_gate_fails_when_there_is_no_signal(dctConfig):
    """With no signal the model can't beat the baseline by the margin, so the gate fails."""
    dfXTrain, _, serYTrain, _ = tplTrainRows(dctConfig, 0.0)
    dctConfig["approval"]["min_auc_gain"] = 0.2
    dctGate = dctCheckModelGate(dfXTrain, serYTrain, dctConfig)
    assert dctGate["passed"] is False


def test_the_required_gain_comes_from_the_config(dctConfig):
    """Raising the threshold in the config flips a pass into a fail."""
    dfXTrain, _, serYTrain, _ = tplTrainRows(dctConfig, 1.0)
    dctConfig["approval"]["min_auc_gain"] = 0.01
    assert dctCheckModelGate(dfXTrain, serYTrain, dctConfig)["passed"] is True
    dctConfig["approval"]["min_auc_gain"] = 0.49
    assert dctCheckModelGate(dfXTrain, serYTrain, dctConfig)["passed"] is False


def test_switching_the_gate_off_passes_without_scoring(dctConfig):
    """When approval.enabled is false the gate passes and reports no scores."""
    dfXTrain, _, serYTrain, _ = tplTrainRows(dctConfig, 0.0)
    dctConfig["approval"]["enabled"] = False
    dctGate = dctCheckModelGate(dfXTrain, serYTrain, dctConfig)
    assert dctGate["passed"] is True
    assert dctGate["model_cv_auc"] is None


def test_model_card_describes_the_model_and_the_gate(dctConfig, tmp_path):
    """The card names the inputs, the gate result and the intended use, and saves as JSON."""
    dfXTrain, _, serYTrain, _ = tplTrainRows(dctConfig, 2.0)
    dctGate = dctCheckModelGate(dfXTrain, serYTrain, dctConfig)
    pthData = tmp_path / "data.csv"
    pthData.write_text("a,b\n1,2\n")
    dctCard = dctBuildModelCard(dctConfig, dctGate, dfXTrain, serYTrain, pthData, tmp_path / "missing.joblib")
    assert dctCard["input_columns"] == ["feature_a", "feature_b"]
    assert dctCard["gate"]["passed"] is True
    assert dctCard["intended_use"] == "Test use."
    assert dctCard["data_fingerprint"] != "missing"
    assert "approve" in dctCard["status"]

    pthSaved = pthSaveModelCard(dctCard, tmp_path / "sub" / "card.json")
    assert json.loads(pthSaved.read_text(encoding="utf-8"))["training_rows"] == len(dfXTrain)


def test_rule_benchmark_flags_rows_at_or_above_the_threshold():
    """The rule says positive where the column is at or above the threshold and learns nothing."""
    dfX = pd.DataFrame({"age": [30, 64, 65, 90]})
    mdlRule = RuleBenchmark(column="age", threshold=65).fit(dfX, pd.Series([0, 0, 1, 1]))
    assert list(mdlRule.predict(dfX)) == [0, 0, 1, 1]
    assert mdlRule.predict_proba(dfX)[:, 1].tolist() == [0, 0, 1, 1]


def test_rule_benchmark_needs_its_column():
    """A rule on a column that isn't an input stops with a clear error."""
    with pytest.raises(ValueError, match="age"):
        RuleBenchmark(column="age", threshold=65).fit(pd.DataFrame({"x": [1, 2]}), pd.Series([0, 1]))


def test_build_model_makes_a_rule_from_settings():
    """mdlBuildModel builds the rule type from the config settings."""
    mdlRule = mdlBuildModel({"type": "rule", "column": "age", "threshold": 70}, 42)
    assert mdlRule.column == "age" and mdlRule.threshold == 70


def test_comparison_includes_the_rule_between_baseline_and_candidates(dctConfig):
    """The table lists the baseline, the benchmark rule, then the candidates."""
    dfXTrain, dfXTest, serYTrain, serYTest = tplTrainRows(dctConfig, 1.5)
    dfResults = dfCompareModels(dfXTrain, dfXTest, serYTrain, serYTest, dctConfig)
    assert list(dfResults["role"]) == ["baseline", "benchmark", "candidate", "candidate"]
    assert list(dfResults["model"])[1] == "rule_a"
    # The rule must get a real cross-validated AUC, not NaN from a failed scoring step.
    assert not dfResults["cv_auc_mean"].isna().any()


def test_evidence_record_names_the_model_the_rule_and_the_limits(dctConfig):
    """The record gives the recommended candidate, the rule verdict, the limits and the recommendation."""
    dfXTrain, dfXTest, serYTrain, serYTest = tplTrainRows(dctConfig, 1.5)
    dfResults = dfCompareModels(dfXTrain, dfXTest, serYTrain, serYTest, dctConfig)
    strRecord = strBuildEvidenceRecord(dfResults, dctConfig)
    assert "Recommended candidate" in strRecord
    assert "rule 'rule_a'" in strRecord
    assert "Test data only." in strRecord
    assert "Do not use." in strRecord
    assert "| baseline | baseline |" in strRecord
