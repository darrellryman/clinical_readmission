"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    test_tracking.py

    Tests for src/tracking.py: file fingerprints, and recording runs in MLflow.
"""

import sys

import pytest

from src.tracking import bLogRun, strFileFingerprint


def test_fingerprint_is_stable_and_changes_with_the_file(tmp_path):
    """The same contents give the same fingerprint and different contents give a different one."""
    pthA = tmp_path / "a.txt"
    pthA.write_text("one")
    strFirst = strFileFingerprint(pthA)
    assert strFirst == strFileFingerprint(pthA)
    pthA.write_text("two")
    assert strFileFingerprint(pthA) != strFirst


def test_fingerprint_of_a_missing_file_says_missing(tmp_path):
    """A file that isn't there is reported, not an error."""
    assert strFileFingerprint(tmp_path / "nope.csv") == "missing"


def test_tracking_off_records_nothing_and_returns_false(dctConfig, capsys):
    """With tracking.enabled false, the call does nothing and says so."""
    assert bLogRun(dctConfig, "train", "x") is False
    assert "switched off" in capsys.readouterr().out


def test_missing_mlflow_package_does_not_stop_the_stage(dctConfig, monkeypatch, capsys):
    """If mlflow can't be imported, the call returns False with a message and doesn't raise."""
    dctConfig["tracking"]["enabled"] = True
    # Setting a module to None in sys.modules makes "import mlflow" fail.
    monkeypatch.setitem(sys.modules, "mlflow", None)
    assert bLogRun(dctConfig, "train", "x") is False
    assert "not installed" in capsys.readouterr().out


def test_a_run_is_recorded_with_params_metrics_and_tags(dctConfig, tmp_path, monkeypatch):
    """A real MLflow run holds the settings, the score and the stage tag that were passed in."""
    mlflow = pytest.importorskip("mlflow")
    # MLflow keeps artefacts in ./mlruns, so work inside a temporary folder.
    monkeypatch.chdir(tmp_path)
    dctConfig["tracking"].update({"enabled": True, "uri": f"sqlite:///{tmp_path / 'm.db'}", "experiment_name": "t"})
    pthFile = tmp_path / "model.txt"
    pthFile.write_text("model")

    assert bLogRun(dctConfig, "train", "run_a", {"C": 1.0}, {"auc": 0.7, "bad": float("nan")}, [pthFile]) is True

    mlflow.set_tracking_uri(dctConfig["tracking"]["uri"])
    serRun = mlflow.search_runs(experiment_names=["t"]).iloc[0]
    assert serRun["params.C"] == "1.0"
    assert serRun["metrics.auc"] == 0.7
    assert serRun["tags.stage"] == "train"
    # A NaN score is skipped, not recorded.
    assert "metrics.bad" not in serRun.index or serRun["metrics.bad"] != serRun["metrics.bad"]
