"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    test_train.py
    
    Tests for the model stages: training, saving and loading, predicting, and the
    evaluation metrics (src/train.py, src/predict.py and src/evaluate.py).
"""

import pandas as pd
import pytest

from src.evaluate import dctComputeMetrics
from src.features import tplBuildFeatures
from src.predict import dfPredict
from src.train import mdlBuildModel, mdlLoadModel, pthSaveModel, mdlTrainModel


def test_train_and_predict(dfSample, dctConfig):
    """A trained model can score data, and the output gains a prediction column."""
    dfX, serY = tplBuildFeatures(dfSample, dctConfig)
    mdlModel = mdlTrainModel(dfX, serY, dctConfig)
    dfResult = dfPredict(mdlModel, dfSample, dctConfig)
    assert "prediction" in dfResult.columns
    assert len(dfResult) == len(dfSample)


def test_model_settings_come_from_the_config(dfSample, dctConfig):
    """The model is built with the values in the config's model section."""
    dctConfig["model"]["max_iter"] = 250
    dctConfig["model"]["C"] = 0.5
    dfX, serY = tplBuildFeatures(dfSample, dctConfig)
    mdlModel = mdlTrainModel(dfX, serY, dctConfig)
    assert mdlModel.max_iter == 250
    assert mdlModel.C == 0.5


def test_predict_rejects_missing_feature_columns(dfSample, dctConfig):
    """Predicting on data without the configured feature columns raises a clear error."""
    dfX, serY = tplBuildFeatures(dfSample, dctConfig)
    mdlModel = mdlTrainModel(dfX, serY, dctConfig)
    with pytest.raises(ValueError, match="feature_b"):
        dfPredict(mdlModel, dfSample.drop(columns=["feature_b"]), dctConfig)


def test_model_round_trip(dfSample, dctConfig, tmp_path):
    """A model saved to disk and loaded back gives the same predictions.

    tmp_path is a pytest fixture that supplies a throwaway folder, so the test
    never writes into the real artifacts/ folder.
    """
    dfX, serY = tplBuildFeatures(dfSample, dctConfig)
    mdlModel = mdlTrainModel(dfX, serY, dctConfig)
    pthFile = pthSaveModel(mdlModel, tmp_path / "model.joblib")
    mdlLoaded = mdlLoadModel(pthFile)
    assert list(mdlLoaded.predict(dfX)) == list(mdlModel.predict(dfX))


def test_load_model_missing_file(tmp_path):
    """Loading a model that doesn't exist raises a clear error."""
    with pytest.raises(FileNotFoundError):
        mdlLoadModel(tmp_path / "nope.joblib")


def test_compute_metrics_perfect_score(dctConfig):
    """Predictions that match the real answers exactly score 1.0 on both metrics."""
    dctMetrics = dctComputeMetrics([0, 1, 1, 0], [0, 1, 1, 0], dctConfig)
    assert dctMetrics["accuracy"] == 1.0
    assert dctMetrics["f1"] == 1.0


def test_compute_metrics_includes_precision_and_recall(dctConfig):
    """Precision and recall are worked out for the positive value set in the config."""
    # 4 real positives. The model finds 2 of them and wrongly flags 1 negative.
    lstTrue = [1, 1, 1, 1, 0, 0, 0, 0]
    lstPred = [1, 1, 0, 0, 1, 0, 0, 0]
    dctMetrics = dctComputeMetrics(lstTrue, lstPred, dctConfig)
    assert dctMetrics["recall"] == 0.5          # caught 2 of 4
    assert abs(dctMetrics["precision"] - 2 / 3) < 1e-9   # 2 of the 3 flagged were right


def test_recall_is_zero_when_the_model_never_predicts_the_positive_value(dctConfig):
    """Always guessing 0 looks accurate on uneven data, and recall exposes it."""
    lstTrue = [1] + [0] * 9
    lstPred = [0] * 10
    dctMetrics = dctComputeMetrics(lstTrue, lstPred, dctConfig)
    assert dctMetrics["accuracy"] == 0.9
    assert dctMetrics["recall"] == 0.0
    assert dctMetrics["precision"] == 0.0


def test_auc_is_added_only_when_probabilities_are_given(dctConfig):
    """roc_auc appears with probabilities, and a perfect ranking scores 1.0."""
    assert "roc_auc" not in dctComputeMetrics([0, 1, 1, 0], [0, 1, 1, 0], dctConfig)
    dctMetrics = dctComputeMetrics([0, 1, 1, 0], [0, 1, 1, 0], dctConfig, [0.1, 0.9, 0.8, 0.2])
    assert dctMetrics["roc_auc"] == 1.0


def test_positive_label_setting_changes_recall(dctConfig):
    """Choosing 0 as the positive value scores the other class."""
    lstTrue = [1, 1, 0, 0]
    lstPred = [1, 1, 1, 0]
    assert dctComputeMetrics(lstTrue, lstPred, dctConfig)["recall"] == 1.0
    dctConfig["evaluation"]["positive_label"] = 0
    assert dctComputeMetrics(lstTrue, lstPred, dctConfig)["recall"] == 0.5


def test_class_weight_setting_reaches_the_model(dfSample, dctConfig):
    """model.class_weight in the config is the one the model is built with."""
    dctConfig["model"]["class_weight"] = "balanced"
    dfX, serY = tplBuildFeatures(dfSample, dctConfig)
    assert mdlTrainModel(dfX, serY, dctConfig).class_weight == "balanced"


def test_balanced_weights_make_the_model_flag_the_rare_class(dctConfig):
    """On uneven data, plain weighting predicts all zeros and balanced weighting finds some 1s."""
    # 5 percent positives, with a weak signal in feature_a.
    dfData = pd.DataFrame({
        "feature_a": [0.0] * 90 + [0.3] * 5 + [0.6] * 5,
        "feature_b": [0.0] * 100,
        "target": [0] * 90 + [0] * 3 + [1] * 2 + [1] * 5,
    })
    dfX, serY = tplBuildFeatures(dfData, dctConfig)

    dctConfig["model"]["class_weight"] = None
    intPlain = int(mdlTrainModel(dfX, serY, dctConfig).predict(dfX).sum())
    dctConfig["model"]["class_weight"] = "balanced"
    intBalanced = int(mdlTrainModel(dfX, serY, dctConfig).predict(dfX).sum())
    assert intPlain == 0
    assert intBalanced > intPlain


def test_prediction_uses_the_same_engineered_columns_as_training(dctConfig):
    """Predicting on raw rows (text sex, no pulse pressure) works after training on engineered ones."""
    dctConfig["features"]["feature_columns"] = ["sys", "dia", "sex"]
    dctConfig["features"]["difference_columns"] = {"pulse_pressure": ["sys", "dia"]}
    dctConfig["features"]["categorical_columns"] = {"sex": ["Female", "Male"]}
    dfData = pd.DataFrame({
        "sys": [110, 120, 130, 140, 150, 160, 115, 125, 135, 145, 155, 165],
        "dia": [70, 75, 80, 85, 90, 95, 72, 78, 82, 88, 92, 98],
        "sex": ["Female", "Male"] * 6,
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    dfX, serY = tplBuildFeatures(dfData, dctConfig)
    mdlModel = mdlTrainModel(dfX, serY, dctConfig)
    dfOut = dfPredict(mdlModel, dfData[["sys", "dia", "sex"]], dctConfig)
    assert len(dfOut) == len(dfData)
    assert "prediction" in dfOut.columns


def test_unknown_model_type_raises(dctConfig):
    """A model type the code doesn't know stops the run."""
    with pytest.raises(ValueError, match="type"):
        mdlBuildModel({"type": "magic"}, 42)


def test_confusion_counts_split_right_and_wrong_predictions(dctConfig):
    """The four counts match a hand-counted example and add up to the number of rows."""
    from src.evaluate import dfConfusionCounts
    dfCounts = dfConfusionCounts([1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 1, 0, 0, 0], dctConfig)
    dctRow = dfCounts.iloc[0].to_dict()
    assert dctRow == {"true_positive": 2, "false_negative": 1, "false_positive": 1, "true_negative": 3}


def test_groups_are_built_from_plain_and_binned_columns(dctConfig):
    """Sex becomes one group per value and age becomes bands named from the config edges."""
    from src.evaluate import dfBuildGroups
    dfData = pd.DataFrame({"sex": ["Female", "Male", "Male"], "age": [0, 45, 90]})
    dctConfig["evaluation"]["group_columns"] = ["sex"]
    dctConfig["evaluation"]["binned_columns"] = {"age": [0, 40, 60, 120]}
    dfGroups = dfBuildGroups(dfData, dfData[["age"]], dctConfig)
    assert list(dfGroups["sex"]) == ["Female", "Male", "Male"]
    assert list(dfGroups["age"]) == ["0-40", "40-60", "60-120"]


def test_scores_by_group_report_each_group(dctConfig):
    """Each group gets its own row with its size and recall."""
    from src.evaluate import dfMetricsByGroup
    dfGroups = pd.DataFrame({"sex": ["F", "F", "F", "M", "M", "M"]})
    dfOut = dfMetricsByGroup(dfGroups, [1, 0, 0, 1, 1, 0], [1, 0, 0, 0, 0, 0], [0.9, 0.2, 0.1, 0.4, 0.3, 0.2], dctConfig)
    dfIndexed = dfOut.set_index("group")
    assert dfIndexed.loc["sex=F", "rows"] == 3
    assert dfIndexed.loc["sex=F", "recall"] == 1.0
    assert dfIndexed.loc["sex=M", "recall"] == 0.0


def test_group_with_one_outcome_has_blank_auc(dctConfig):
    """AUC can't be worked out when a group has only one kind of outcome, so it is left blank."""
    from src.evaluate import dfMetricsByGroup
    dfGroups = pd.DataFrame({"sex": ["F", "F"]})
    dfOut = dfMetricsByGroup(dfGroups, [0, 0], [0, 0], [0.1, 0.2], dctConfig)
    assert pd.isna(dfOut.loc[0, "roc_auc"])


def test_saved_features_show_the_encoded_columns_and_the_split(dfSample, dctConfig, tmp_path):
    """The inspection file holds the inputs, the target and a train or test label for every row."""
    from src.features import lstSplitData
    from src.train import pthSaveFeatures
    dfX, serY = tplBuildFeatures(dfSample, dctConfig)
    dfXTrain, dfXTest, serYTrain, serYTest = lstSplitData(dfX, serY, dctConfig)
    pthFile = pthSaveFeatures(dfXTrain, dfXTest, serYTrain, serYTest, tmp_path / "f.csv")
    dfSaved = pd.read_csv(pthFile)
    assert list(dfSaved.columns) == list(dfX.columns) + [serY.name, "split"]
    assert len(dfSaved) == len(dfSample)
    assert (dfSaved["split"] == "train").sum() == len(dfXTrain)


def test_saved_features_include_the_encoded_sex_and_pulse_pressure_columns(tmp_path, dctConfig):
    """With the rules switched on, the saved file has sex_Male and pulse_pressure, not the text column."""
    from src.features import lstSplitData
    from src.train import pthSaveFeatures
    dctConfig["features"]["feature_columns"] = ["sys", "dia", "sex"]
    dctConfig["features"]["difference_columns"] = {"pulse_pressure": ["sys", "dia"]}
    dctConfig["features"]["categorical_columns"] = {"sex": ["Female", "Male"]}
    dfData = pd.DataFrame({
        "sys": range(100, 140), "dia": range(60, 100), "sex": ["Female", "Male"] * 20,
        "target": [1 if x % 5 == 0 else 0 for x in range(40)],
    })
    dfX, serY = tplBuildFeatures(dfData, dctConfig)
    dfXTrain, dfXTest, serYTrain, serYTest = lstSplitData(dfX, serY, dctConfig)
    dfSaved = pd.read_csv(pthSaveFeatures(dfXTrain, dfXTest, serYTrain, serYTest, tmp_path / "f.csv"))
    assert "sex_Male" in dfSaved.columns and "pulse_pressure" in dfSaved.columns
    assert "sex" not in dfSaved.columns
    assert set(dfSaved["sex_Male"]) == {0, 1}
