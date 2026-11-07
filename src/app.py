"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    app.py

    The prediction service: the saved model behind a small web API (FastAPI).

    It loads the model once when it starts, then answers four requests:
        GET  /         A web page with a form. Enter one patient's predictors and
                       it answers "Possible readmission after 30 days: Yes" or "No".
        GET  /health   Is the service up, and is the model loaded?
        GET  /info     The model card: what the model is for, what built it, the gate result.
        POST /predict  Score one or more patients.

    The request format is built from config.yaml (features.feature_columns and
    features.categorical_columns), so the API contract can't drift from the
    model's inputs. A patient with a missing field, an extra field, a value of
    the wrong type or a category that isn't on the config list is refused with
    a 422 error before it reaches the model. Accepted patients go to the saved
    pipeline, which builds the engineered columns itself, as it did in training.

    The service is for demonstration. The model is trained on synthetic data and
    must not be used to make decisions about real patients.

    Run from the project root (needs: pip install -r requirements.txt):
        python -m src.app
    then open http://127.0.0.1:8000/ for the form, or http://127.0.0.1:8000/docs
    to try the API directly.
"""

import html
import json
from pathlib import Path
from typing import Literal

# FastAPI builds the web service. BaseModel, ConfigDict, Field and create_model
# (from pydantic) describe the shape of a request and check it automatically.
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

import pandas as pd

from src.config import dctLoadConfig
from src.features import dfRawInputs
from src.train import mdlLoadModel


class PredictionOut(BaseModel):
    """The answer for one patient."""

    prediction: int = Field(description="The predicted class. 1 means readmission within 30 days is predicted.")
    probability: float = Field(description="The model's estimated chance of the positive class, from 0 to 1.")


class PredictResponse(BaseModel):
    """The answer for a request: one prediction per patient, in the order sent."""

    predictions: list[PredictionOut]


def objBuildPatientModel(dctConfig: dict):
    """Build the request model for one patient from config.yaml.

    Every feature column becomes a required field. A categorical column (such as
    sex) accepts only the categories on its config list. Every other column is a
    number. Extra fields are refused.

    Args:
        dctConfig: The loaded config.yaml settings. Uses features.feature_columns
            and features.categorical_columns.

    Returns:
        A pydantic model class describing one patient.
    """
    dctCategories = dctConfig["features"]["categorical_columns"] or {}
    dctFields = {}
    for strColumn in dctConfig["features"]["feature_columns"]:
        if strColumn in dctCategories:
            # Literal[...] allows only these exact values, so an unknown category is refused.
            dctFields[strColumn] = (Literal[tuple(dctCategories[strColumn])], ...)
        else:
            dctFields[strColumn] = (float, ...)
    return create_model("Patient", __config__=ConfigDict(extra="forbid"), **dctFields)


def strBuildFormPage(dctConfig: dict) -> str:
    """Build the web page with one input for each predictor, from config.yaml.

    A categorical column (such as sex) becomes a drop-down list of its config
    categories. Every other feature column becomes a number box. The boxes start
    with the values in service.form_defaults (typical healthy adult values). Because the
    form is built from features.feature_columns and features.categorical_columns,
    it can't drift from the model's inputs. The page sends the values to /predict
    and shows "Possible readmission after 30 days: Yes" or "No".

    Args:
        dctConfig: The loaded config.yaml settings. Uses features.feature_columns,
            features.categorical_columns, service.form_defaults and service.header_image.

    Returns:
        The page as one HTML string.
    """
    dctCategories = dctConfig["features"]["categorical_columns"] or {}
    # The banner at the top is optional. It is shown only if service.header_image names a real file.
    strHeaderPath = dctConfig["service"].get("header_image")
    blnHeader = bool(strHeaderPath) and Path(strHeaderPath).is_file()
    strHeader = ('<img class="banner" src="/header-image" '
                 'alt="Healthcare banner">') if blnHeader else ""
    # The starting values come from service.form_defaults. A column with none starts blank.
    dctDefaults = dctConfig["service"].get("form_defaults") or {}
    lstRows = []
    for strColumn in dctConfig["features"]["feature_columns"]:
        strName = html.escape(strColumn)
        if strColumn in dctCategories:
            strOptions = "".join(
                f'<option value="{html.escape(str(varCategory))}"'
                f'{" selected" if strColumn in dctDefaults and varCategory == dctDefaults[strColumn] else ""}>'
                f'{html.escape(str(varCategory))}</option>'
                for varCategory in dctCategories[strColumn]
            )
            strInput = f'<select name="{strName}" data-kind="text" required>{strOptions}</select>'
        else:
            strValue = f' value="{html.escape(str(dctDefaults[strColumn]))}"' if strColumn in dctDefaults else ""
            strInput = f'<input name="{strName}" data-kind="number" type="number" step="any"{strValue} required>'
        lstRows.append(f'<label>{strName}{strInput}</label>')

    return """<!doctype html>
<html lang="en-AU">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>30-day readmission check</title>
<style>
  body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto 2rem; padding: 0 1rem; color: #222; }
  .banner { display: block; width: 100%; height: auto; }
  main { max-width: 520px; margin: 1.5rem auto 0; }
  label { display: flex; justify-content: space-between; align-items: center; margin: .5rem 0; }
  input, select { width: 55%; padding: .35rem; box-sizing: border-box; }
  button { margin-top: 1rem; padding: .6rem 1.2rem; font-size: 1rem; cursor: pointer; }
  #answer { margin-top: 1.2rem; padding: 1rem; border-radius: 6px; font-weight: bold; display: none; }
  #answer.yes { background: #fde8e8; border: 1px solid #c0392b; }
  #answer.no { background: #e8f5e9; border: 1px solid #2e7d32; }
  #answer.error { background: #fff3cd; border: 1px solid #b58900; font-weight: normal; }
  small { color: #666; }
</style>
</head>
<body>
""" + strHeader + """
<main>
<h1>30-day readmission check</h1>
<p><small>Demonstration only. The model is trained on synthetic data and must not be used for real patients.</small></p>
<form id="form">
""" + "\n".join(lstRows) + """
<button type="submit">Predict</button>
</form>
<div id="answer"></div>
</main>
<script>
document.getElementById("form").addEventListener("submit", async function (objEvent) {
  objEvent.preventDefault();
  const objPatient = {};
  for (const objField of this.elements) {
    if (!objField.name) continue;
    objPatient[objField.name] = objField.dataset.kind === "number" ? parseFloat(objField.value) : objField.value;
  }
  const objBox = document.getElementById("answer");
  try {
    const objResponse = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({patients: [objPatient]})
    });
    if (!objResponse.ok) {
      objBox.className = "error";
      objBox.textContent = "The service could not score this patient. Check the values and try again.";
    } else {
      const objResult = (await objResponse.json()).predictions[0];
      const blnYes = objResult.prediction === 1;
      objBox.className = blnYes ? "yes" : "no";
      objBox.textContent = "Possible readmission after 30 days: " + (blnYes ? "Yes" : "No");
    }
  } catch (objError) {
    objBox.className = "error";
    objBox.textContent = "Could not reach the service.";
  }
  objBox.style.display = "block";
});
</script>
</body>
</html>"""


def objBuildApp(dctConfig: dict) -> FastAPI:
    """Build the web service around the saved model.

    Args:
        dctConfig: The loaded config.yaml settings. Uses model.path,
            approval.model_card_path, features, evaluation.positive_label and
            service.max_patients.

    Returns:
        A FastAPI application, ready to serve.

    Raises:
        FileNotFoundError: If there is no saved model. Run src.train first.
    """
    # Load the model once, here, not on every request. This fails at start-up if
    # the model is missing, which is better than failing on the first patient.
    mdlModel = mdlLoadModel(dctConfig["model"]["path"])
    lstFeatureColumns = dctConfig["features"]["feature_columns"]
    varPositive = dctConfig["evaluation"]["positive_label"]
    intPositive = list(mdlModel.classes_).index(varPositive)

    clsPatient = objBuildPatientModel(dctConfig)
    clsRequest = create_model(
        "PredictRequest",
        patients=(list[clsPatient], Field(min_length=1, max_length=dctConfig["service"]["max_patients"])),
    )

    objApp = FastAPI(
        title="Readmission prediction service",
        description="Scores patients for 30-day readmission risk. Demonstration only, trained on synthetic data.",
    )

    @objApp.get("/", response_class=HTMLResponse, include_in_schema=False)
    def strHomePage() -> str:
        """Show the form for entering a patient's predictors."""
        return strBuildFormPage(dctConfig)

    @objApp.get("/header-image", include_in_schema=False)
    def objHeaderImage() -> FileResponse:
        """Send the banner picture shown at the top of the form page."""
        pthImage = Path(dctConfig["service"].get("header_image") or "")
        if not pthImage.is_file():
            raise HTTPException(status_code=404, detail="No header image is set.")
        return FileResponse(pthImage)

    @objApp.get("/health")
    def dctHealth() -> dict:
        """Report that the service is up and the model is loaded."""
        return {"status": "ok", "model_loaded": True}

    @objApp.get("/info")
    def dctInfo() -> dict:
        """Return the model card, or a note that there isn't one."""
        pthCard = Path(dctConfig["approval"]["model_card_path"])
        if not pthCard.is_file():
            return {"model_card": None, "note": "No model card found. Run src.train."}
        return json.loads(pthCard.read_text(encoding="utf-8"))

    @objApp.post("/predict", response_model=PredictResponse)
    def objPredict(objRequest: clsRequest) -> PredictResponse:
        """Score the patients in the request.

        Args:
            objRequest: A body with a list called patients. Each patient has every
                feature column.

        Returns:
            One prediction and probability per patient, in the order sent.
        """
        # One row per patient, in the config's column order. The saved pipeline
        # builds the engineered columns itself, the same way training did.
        dfRaw = dfRawInputs(pd.DataFrame([objPatient.model_dump() for objPatient in objRequest.patients]), dctConfig)
        arrPredictions = mdlModel.predict(dfRaw)
        arrProbabilities = mdlModel.predict_proba(dfRaw)[:, intPositive]
        return PredictResponse(predictions=[
            PredictionOut(prediction=int(intPrediction), probability=round(float(fltProbability), 4))
            for intPrediction, fltProbability in zip(arrPredictions, arrProbabilities)
        ])

    return objApp


def objCreateApp() -> FastAPI:
    """Build the service from config.yaml. Used by uvicorn, which calls it with no arguments."""
    return objBuildApp(dctLoadConfig())


def subMain() -> None:
    """Command line entry point: start the service on the host and port in config.yaml."""
    # Imported here so the rest of the pipeline doesn't need uvicorn installed.
    import uvicorn

    dctConfig = dctLoadConfig()
    uvicorn.run(objBuildApp(dctConfig), host=dctConfig["service"]["host"], port=dctConfig["service"]["port"])


if __name__ == "__main__":
    subMain()
