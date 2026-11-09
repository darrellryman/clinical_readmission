"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    test_monitor.py

    Tests for src/monitor.py: the PASS, WATCH and ALERT results, the PSI sums,
    the data quality counts, and a full run on a normal and a shifted batch.
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src import monitor
from src.config import ConfigError, subCheckConfig


@pytest.fixture
def dctMonConfig(dctConfig) -> dict:
    """The shared test config, with a limit on feature_a and a category rule on feature_b."""
    dctConfig["validation"]["column_ranges"] = {"feature_a": {"min": 0, "max": 100}}
    return dctConfig


def test_status_thresholds():
    """Below the watch level passes, the watch level watches, the alert level alerts."""
    assert monitor.strStatus(0.05, 0.1, 0.25) == "PASS"
    assert monitor.strStatus(0.1, 0.1, 0.25) == "WATCH"
    assert monitor.strStatus(0.3, 0.1, 0.25) == "ALERT"


def test_worst_picks_the_highest_ranked_result():
    """One ALERT outranks any number of PASS results, and an empty list is a PASS."""
    assert monitor.strWorst(["PASS", "WATCH", "PASS"]) == "WATCH"
    assert monitor.strWorst(["PASS", "ALERT", "WATCH"]) == "ALERT"
    assert monitor.strWorst([]) == "PASS"


def test_psi_is_zero_for_identical_shares_and_large_for_a_big_shift():
    """The same shares give 0. A big move between bins gives a PSI well above 0.25."""
    assert monitor.fltPsiFromShares([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert monitor.fltPsiFromShares([0.5, 0.5], [0.9, 0.1]) > 0.25


def test_empty_bin_does_not_break_the_psi():
    """A bin that is empty in the batch gives a finite number, not infinity or NaN."""
    fltPsi = monitor.fltPsiFromShares([0.5, 0.5], [1.0, 0.0])
    assert np.isfinite(fltPsi)


def test_numeric_psi_same_sample_is_small_and_shifted_sample_is_large():
    """A sample compared with itself is stable. The same sample moved far to the right is not."""
    serRef = pd.Series(np.linspace(0, 100, 500))
    assert monitor.fltNumericPsi(serRef, serRef.copy(), 10) < 0.01
    assert monitor.fltNumericPsi(serRef, serRef + 80, 10) > 0.25


def test_numeric_psi_handles_a_column_that_never_changes():
    """A constant training column is compared as same value or not, without an error."""
    serRef = pd.Series([5.0] * 50)
    assert monitor.fltNumericPsi(serRef, pd.Series([5.0] * 20), 10) < 0.01
    assert monitor.fltNumericPsi(serRef, pd.Series([9.0] * 20), 10) > 0.25


def test_category_psi_sees_a_changed_mix():
    """A 50/50 text column that becomes 95/5 is flagged."""
    serRef = pd.Series(["Female", "Male"] * 100)
    serNew = pd.Series(["Female"] * 95 + ["Male"] * 5)
    assert monitor.fltCategoryPsi(serRef, serRef.copy()) < 0.01
    assert monitor.fltCategoryPsi(serRef, serNew) > 0.25


def test_input_drift_gives_one_row_per_feature(dctMonConfig, dfSample):
    """Each configured feature gets a PSI and a result. Same data passes."""
    dfResult = monitor.dfInputDrift(dfSample, dfSample.copy(), dctMonConfig)
    assert list(dfResult["column"]) == dctMonConfig["features"]["feature_columns"]
    assert set(dfResult["result"]) == {"PASS"}


def test_out_of_range_counts_values_outside_the_limits(dctMonConfig, dfSample):
    """Two values above the limit are counted, with their share of the batch."""
    dfBatch = dfSample.copy()
    dfBatch.loc[[0, 1], "feature_a"] = 500
    dfResult = monitor.dfOutOfRange(dfBatch, dctMonConfig)
    dfRow = dfResult[dfResult["column"] == "feature_a"].iloc[0]
    assert dfRow["count"] == 2
    assert dfRow["share"] == pytest.approx(2 / len(dfBatch))


def test_out_of_range_counts_unseen_categories(dctMonConfig, dfSample):
    """A category not in features.categorical_columns is counted."""
    dctMonConfig["features"]["categorical_columns"] = {"feature_c": ["x", "y"]}
    dfBatch = dfSample.copy()
    dfBatch["feature_c"] = ["x"] * 18 + ["z"] * 2
    dfResult = monitor.dfOutOfRange(dfBatch, dctMonConfig)
    dfRow = dfResult[dfResult["check"] == "unseen category"].iloc[0]
    assert dfRow["count"] == 2


def test_data_quality_flags_missing_values(dctMonConfig, dfSample):
    """Three missing values in 20 rows (15 percent) is an ALERT with the default thresholds."""
    dfBatch = dfSample.copy()
    dfBatch.loc[[0, 1, 2], "feature_b"] = None
    dfResult = monitor.dfDataQuality(dfBatch, dctMonConfig)
    dfRow = dfResult[(dfResult["column"] == "feature_b") & (dfResult["check"] == "missing")].iloc[0]
    assert dfRow["count"] == 3
    assert dfRow["result"] == "ALERT"


def test_usable_rows_drop_missing_and_unseen_categories(dctMonConfig, dfSample):
    """Rows the model could not score are left out of the flag rate."""
    dctMonConfig["features"]["categorical_columns"] = {"feature_c": ["x", "y"]}
    dfBatch = dfSample.copy()
    dfBatch["feature_c"] = ["x"] * 20
    dfBatch.loc[0, "feature_a"] = None
    dfBatch.loc[1, "feature_c"] = "z"
    assert len(monitor.dfUsableRows(dfBatch, dctMonConfig)) == 18


def pthMakeMonitorRun(dctConfig, dfSample, tmp_path, dfBatch):
    """Write a reference file, a batch file and a saved model, and point the config at them."""
    pthRef = tmp_path / "clean.csv"
    pthBatch = tmp_path / "batch.csv"
    pthModel = tmp_path / "model.joblib"
    dfSample.to_csv(pthRef, index=False)
    dfBatch.drop(columns=["target"], errors="ignore").to_csv(pthBatch, index=False)
    mdlModel = DummyClassifier(strategy="most_frequent").fit(dfSample[["feature_a", "feature_b"]], dfSample["target"])
    joblib.dump(mdlModel, pthModel)
    dctConfig["data"]["processed_path"] = str(pthRef)
    return str(pthBatch), str(pthModel)


def test_monitor_passes_a_batch_that_looks_like_training(dctConfig, dfSample, tmp_path):
    """The training data checked against itself is a PASS overall."""
    strBatch, strModel = pthMakeMonitorRun(dctConfig, dfSample, tmp_path, dfSample.copy())
    assert monitor.subMonitorAll(dctConfig, strBatch, strModel) == "PASS"


def test_monitor_alerts_on_a_shifted_batch(dctConfig, dfSample, tmp_path):
    """A batch whose inputs have moved far from training is an ALERT."""
    dfShifted = dfSample.copy()
    dfShifted["feature_a"] = dfShifted["feature_a"] + 500
    strBatch, strModel = pthMakeMonitorRun(dctConfig, dfSample, tmp_path, dfShifted)
    assert monitor.subMonitorAll(dctConfig, strBatch, strModel) == "ALERT"


def test_monitor_names_a_missing_feature_column(dctConfig, dfSample, tmp_path):
    """A batch without a feature column stops with a message that names it."""
    strBatch, strModel = pthMakeMonitorRun(dctConfig, dfSample, tmp_path, dfSample.drop(columns=["feature_b"]))
    with pytest.raises(ValueError, match="feature_b"):
        monitor.subMonitorAll(dctConfig, strBatch, strModel)


def test_monitor_alerts_when_nothing_can_be_scored(dctConfig, dfSample, tmp_path):
    """If every row in the batch has a missing input, the output check is an ALERT."""
    dfEmpty = dfSample.copy()
    dfEmpty["feature_a"] = None
    strBatch, strModel = pthMakeMonitorRun(dctConfig, dfSample, tmp_path, dfEmpty)
    assert monitor.subMonitorAll(dctConfig, strBatch, strModel) == "ALERT"


def test_config_rejects_a_watch_level_above_the_alert_level(dctConfig):
    """psi_watch above psi_alert is a config mistake and the message names it."""
    dctConfig["monitoring"]["psi_watch"] = 0.5
    with pytest.raises(ConfigError, match="psi_watch"):
        subCheckConfig(dctConfig)
