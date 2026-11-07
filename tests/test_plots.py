"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    test_plots.py

    Tests for src/plots.py: the evidence graphs.
"""

import numpy as np
import pandas as pd
import pytest

# The tests are skipped if matplotlib is not installed (it is optional).
pytest.importorskip("matplotlib")

from src.config import subCheckConfig
from src.plots import lstDrawAllFigures, lstModelSettings, objPlotConfusion, strColour
from src.train import arrCrossValScores, mdlBuildModel, mdlTrainModel, tplCrossValAuc
from src.features import dfRawInputs, tplBuildFeatures, lstSplitData


def dfMakeData(intRows: int = 200) -> pd.DataFrame:
    """Test helper: feature_a carries some signal, about 20 percent are positive, and there is a group column."""
    objRng = np.random.default_rng(0)
    serTarget = pd.Series((objRng.random(intRows) < 0.2).astype(int))
    return pd.DataFrame({
        "feature_a": objRng.normal(size=intRows) + serTarget * 1.5,
        "feature_b": objRng.normal(size=intRows),
        "target": serTarget,
        "grp": np.where(objRng.random(intRows) < 0.5, "x", "y"),
    })


def test_four_files_are_written_and_not_empty(dctConfig, tmp_path):
    """With a group column set, all four graphs are written as non-empty PNG files."""
    dctConfig["plots"]["folder"] = str(tmp_path)
    dctConfig["evaluation"]["group_columns"] = ["grp"]
    dfData = dfMakeData()
    dfX, serY = tplBuildFeatures(dfData, dctConfig)
    dfXTrain, _, serYTrain, _ = lstSplitData(dfX, serY, dctConfig)
    mdlSaved = mdlTrainModel(dfRawInputs(dfData, dctConfig).loc[dfXTrain.index], serYTrain, dctConfig)
    lstPaths = lstDrawAllFigures(dfData, mdlSaved, dctConfig)
    assert sorted(pth.name for pth in lstPaths) == ["1_roc_pr.png", "2_cv_spread.png", "3_confusion.png", "4_groups.png"]
    for pthFile in lstPaths:
        assert pthFile.stat().st_size > 1000


def test_group_graph_is_skipped_when_no_groups_are_set(dctConfig, tmp_path):
    """With no grouping in the config, only three graphs are written and nothing fails."""
    dctConfig["plots"]["folder"] = str(tmp_path)
    dfData = dfMakeData()
    dfX, serY = tplBuildFeatures(dfData, dctConfig)
    dfXTrain, _, serYTrain, _ = lstSplitData(dfX, serY, dctConfig)
    mdlSaved = mdlTrainModel(dfRawInputs(dfData, dctConfig).loc[dfXTrain.index], serYTrain, dctConfig)
    assert len(lstDrawAllFigures(dfData, mdlSaved, dctConfig)) == 3


def test_model_order_is_baseline_benchmarks_candidates(dctConfig):
    """The graphs list models in the same order as the comparison table."""
    assert [strName for strName, _ in lstModelSettings(dctConfig)] == ["baseline", "rule_a", "logistic_regression", "random_forest"]


def test_palette_repeats_when_there_are_more_models_than_colours(dctConfig):
    """A short palette wraps round instead of raising an error."""
    dctConfig["plots"]["palette"] = ["#111111", "#222222"]
    assert strColour(dctConfig, 2) == "#111111"


def test_fold_scores_agree_with_the_mean_and_spread_used_elsewhere(dctConfig):
    """The per-fold scores on the graph give exactly the mean and SD that compare.py reports."""
    dfX, serY = tplBuildFeatures(dfMakeData(), dctConfig)
    dfXTrain, _, serYTrain, _ = lstSplitData(dfX, serY, dctConfig)
    mdlModel = mdlBuildModel(dctConfig["comparison"]["candidates"]["logistic_regression"], 42)
    arrScores = arrCrossValScores(mdlModel, dfXTrain, serYTrain, 3, 42)
    assert (float(arrScores.mean()), float(arrScores.std())) == tplCrossValAuc(mdlModel, dfXTrain, serYTrain, 3, 42)


def test_confusion_figure_shows_the_counts(dctConfig):
    """The text drawn on the confusion grid holds the four counts it was given."""
    dfCounts = pd.DataFrame([{"true_positive": 7, "false_negative": 3, "false_positive": 11, "true_negative": 99}])
    objFig = objPlotConfusion(dfCounts)
    lstText = [objText.get_text() for objText in objFig.axes[0].texts]
    for strCount in ("7", "3", "11", "99"):
        assert strCount in lstText


def test_config_check_requires_the_plots_section(dctConfig):
    """Dropping the plots section stops the run with a clear config error."""
    del dctConfig["plots"]
    with pytest.raises(Exception, match="plots"):
        subCheckConfig(dctConfig)
