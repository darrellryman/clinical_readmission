"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    test_serve.py

    Tests for src/serve.py: the prediction service. They call the API in memory
    with FastAPI's test client, so no server has to be running.
"""

import json

import pytest

# These tests need the service libraries. Skip them, not fail, if they aren't installed.
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from src.serve import objBuildApp
from src.features import dfRawInputs, tplBuildFeatures
from src.train import mdlTrainModel, pthSaveModel


@pytest.fixture
def objClient(dctConfig, dfSample, tmp_path):
    """A test client for a service built on a small model saved in a temporary folder."""
    dctConfig["model"]["path"] = str(tmp_path / "model.joblib")
    dctConfig["approval"]["model_card_path"] = str(tmp_path / "card.json")
    serY = tplBuildFeatures(dfSample, dctConfig)[1]
    pthSaveModel(mdlTrainModel(dfRawInputs(dfSample, dctConfig), serY, dctConfig), dctConfig["model"]["path"])
    return TestClient(objBuildApp(dctConfig))


def test_health_says_the_model_is_loaded(objClient):
    """GET /health reports ok and that the model loaded."""
    objResponse = objClient.get("/health")
    assert objResponse.status_code == 200
    assert objResponse.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_one_answer_per_patient_in_order(objClient):
    """Two patients in, two predictions out, each with a class and a probability."""
    objResponse = objClient.post("/predict", json={"patients": [
        {"feature_a": 1, "feature_b": 0}, {"feature_a": 18, "feature_b": 2},
    ]})
    assert objResponse.status_code == 200
    lstPredictions = objResponse.json()["predictions"]
    assert len(lstPredictions) == 2
    assert all(0 <= dctOne["probability"] <= 1 for dctOne in lstPredictions)
    # The sample data's target rises with feature_a, so the second patient scores higher.
    assert lstPredictions[1]["probability"] > lstPredictions[0]["probability"]


def test_service_matches_the_model_called_directly(objClient, dctConfig, dfSample):
    """The service gives the same classes as scoring the same rows with the saved model."""
    from src.train import mdlLoadModel
    lstExpected = mdlLoadModel(dctConfig["model"]["path"]).predict(dfRawInputs(dfSample, dctConfig)).tolist()
    lstPatients = dfSample[["feature_a", "feature_b"]].to_dict(orient="records")[:5]
    lstGot = [dctOne["prediction"] for dctOne in objClient.post("/predict", json={"patients": lstPatients}).json()["predictions"]]
    assert lstGot == lstExpected[:5]


def test_a_missing_field_is_refused(objClient):
    """A patient without every feature column gets a 422, not a guess."""
    assert objClient.post("/predict", json={"patients": [{"feature_a": 1}]}).status_code == 422


def test_an_extra_field_is_refused(objClient):
    """A field the model doesn't use is refused, so a typo can't slip through."""
    assert objClient.post("/predict", json={"patients": [{"feature_a": 1, "feature_b": 0, "mortality": 1}]}).status_code == 422


def test_a_wrong_type_is_refused(objClient):
    """Text where a number belongs gets a 422."""
    assert objClient.post("/predict", json={"patients": [{"feature_a": "old", "feature_b": 0}]}).status_code == 422


def test_an_empty_request_is_refused(objClient):
    """No patients at all gets a 422."""
    assert objClient.post("/predict", json={"patients": []}).status_code == 422


def test_too_many_patients_are_refused(objClient, dctConfig):
    """A request over service.max_patients gets a 422."""
    lstPatients = [{"feature_a": 1, "feature_b": 0}] * (dctConfig["service"]["max_patients"] + 1)
    assert objClient.post("/predict", json={"patients": lstPatients}).status_code == 422


def test_an_unknown_category_is_refused(dctConfig, dfSample, tmp_path):
    """A category that isn't on the config list gets a 422 before it reaches the model."""
    dctConfig["features"]["feature_columns"] = ["feature_a", "group"]
    dctConfig["features"]["categorical_columns"] = {"group": ["x", "y"]}
    dfData = dfSample.assign(group=["x", "y"] * 10)[["feature_a", "group", "target"]]
    dctConfig["model"]["path"] = str(tmp_path / "model.joblib")
    serY = tplBuildFeatures(dfData, dctConfig)[1]
    pthSaveModel(mdlTrainModel(dfRawInputs(dfData, dctConfig), serY, dctConfig), dctConfig["model"]["path"])
    objClient = TestClient(objBuildApp(dctConfig))
    assert objClient.post("/predict", json={"patients": [{"feature_a": 1, "group": "x"}]}).status_code == 200
    assert objClient.post("/predict", json={"patients": [{"feature_a": 1, "group": "z"}]}).status_code == 422


def test_info_returns_the_model_card(objClient, dctConfig):
    """GET /info returns the model card file when it exists, and a note when it doesn't."""
    assert objClient.get("/info").json()["model_card"] is None
    with open(dctConfig["approval"]["model_card_path"], "w", encoding="utf-8") as objFile:
        json.dump({"status": "Passed the automated gate."}, objFile)
    assert objClient.get("/info").json()["status"] == "Passed the automated gate."


def test_missing_model_stops_the_service_at_start_up(dctConfig, tmp_path):
    """Building the service without a saved model raises, so the failure is at start-up."""
    dctConfig["model"]["path"] = str(tmp_path / "nothing.joblib")
    with pytest.raises(FileNotFoundError):
        objBuildApp(dctConfig)


def test_home_page_has_a_box_for_every_predictor(objClient, dctConfig):
    """GET / returns a web page with one input for each feature column."""
    objResponse = objClient.get("/")
    assert objResponse.status_code == 200
    assert "text/html" in objResponse.headers["content-type"]
    for strColumn in dctConfig["features"]["feature_columns"]:
        assert f'name="{strColumn}"' in objResponse.text
    assert "Possible readmission after 30 days" in objResponse.text


def test_home_page_starts_with_the_configured_defaults(dctConfig, objClient):
    """service.form_defaults fills the boxes, and a column with none starts blank."""
    dctConfig["service"]["form_defaults"] = {"feature_a": 7}
    objPage = TestClient(objBuildApp(dctConfig)).get("/").text
    assert 'name="feature_a" data-kind="number" type="number" step="any" value="7"' in objPage
    assert 'name="feature_b" data-kind="number" type="number" step="any" required' in objPage


def test_home_page_shows_the_banner_when_the_file_exists(dctConfig, objClient, tmp_path):
    """service.header_image adds the banner and serves the picture. No image means no banner."""
    pthImage = tmp_path / "banner.jpg"
    pthImage.write_bytes(b"not a real picture")
    dctConfig["service"]["header_image"] = str(pthImage)
    objWithBanner = TestClient(objBuildApp(dctConfig))
    assert 'src="/header-image"' in objWithBanner.get("/").text
    assert objWithBanner.get("/header-image").content == b"not a real picture"

    dctConfig["service"]["header_image"] = None
    objWithout = TestClient(objBuildApp(dctConfig))
    assert "header-image" not in objWithout.get("/").text
    assert objWithout.get("/header-image").status_code == 404
