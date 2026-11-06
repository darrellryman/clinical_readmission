"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    test_rtf_output.py

End-to-end test: run every stage and check that each one saves its printed
output to its own RTF file instead of the screen.

The RTF files go to testing.rtf_path from the real config.yaml, which is a
different folder from outputs.rtf_path, so this never overwrites a real run.
Everything else (data, model, metrics) is written to a throwaway folder.
"""

import sys
from pathlib import Path

import yaml

from src import evaluate, ingest, predict, train, validate
from src.config import dctLoadConfig
from src.display import pthResolveRtfPath

# The real config.yaml at the project root, found from this file's location.
PTH_REAL_CONFIG = Path(__file__).resolve().parent.parent / "config.yaml"


def pthMakeRunConfig(dctConfig, dfSample, tmp_path) -> Path:
    """Write sample data and a config file for one test run, and return the config path.

    Only the RTF location is taken from the real config.yaml (testing.rtf_path).
    """
    pthRaw = tmp_path / "raw.csv"
    pthNew = tmp_path / "new.csv"
    dfSample.to_csv(pthRaw, index=False)
    dfSample.drop(columns=["target"]).to_csv(pthNew, index=False)

    dctConfig["data"]["raw_path"] = str(pthRaw)
    dctConfig["data"]["processed_path"] = str(tmp_path / "clean.csv")
    dctConfig["data"]["new_data_path"] = str(pthNew)
    dctConfig["model"]["path"] = str(tmp_path / "model.joblib")
    # Keep the model card in the temporary folder too, so a test run never overwrites a real card.
    dctConfig["approval"]["model_card_path"] = str(tmp_path / "model_card.json")
    dctConfig["outputs"]["metrics_path"] = str(tmp_path / "metrics.json")
    dctConfig["outputs"]["predictions_path"] = str(tmp_path / "predictions.csv")
    dctConfig["outputs"]["rtf_path"] = dctLoadConfig(PTH_REAL_CONFIG)["testing"]["rtf_path"]

    pthConfig = tmp_path / "run.yaml"
    pthConfig.write_text(yaml.safe_dump(dctConfig))
    return pthConfig


def test_every_stage_saves_its_output_to_its_own_rtf_file(dctConfig, dfSample, tmp_path, monkeypatch, capsys):
    """Each stage writes its own RTF file, and its results are not printed to the screen."""
    pthConfig = pthMakeRunConfig(dctConfig, dfSample, tmp_path)
    strTemplate = dctConfig["outputs"]["rtf_path"]

    # The words each stage prints, so we can find them in its file.
    dctExpected = {
        "ingest": ("ingest", ingest, "Saved 20 rows"),
        "validate": ("validate", validate, "20 rows passed validation"),
        "train": ("train", train, "Model saved to"),
        "evaluate": ("evaluate", evaluate, "Metrics written to"),
        "predict": ("predict", predict, "predictions to"),
    }
    for strStage, (_, objModule, strWords) in dctExpected.items():
        monkeypatch.setattr(sys, "argv", [strStage, "--config", str(pthConfig)])
        objModule.subMain()

        strScreen = capsys.readouterr().out
        assert strWords not in strScreen, f"{strStage} still printed its results to the screen"
        assert "Output saved to" in strScreen

        pthRtf = pthResolveRtfPath(strTemplate, strStage)
        assert pthRtf.exists(), f"{strStage} did not write {pthRtf}"
        assert strWords in pthRtf.read_text(encoding="ascii")


def test_the_test_rtf_path_is_not_the_real_run_path(dctConfig):
    """The RTF location used by these tests is not the one a real run writes to."""
    dctReal = dctLoadConfig(PTH_REAL_CONFIG)
    assert dctReal["testing"]["rtf_path"] != dctReal["outputs"]["rtf_path"]
