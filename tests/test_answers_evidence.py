"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    test_answers_evidence.py

    Tests for src/answers_evidence.py, which holds the assessment answers and the
    evidence behind them. The answers no longer sit in the stage files, and one test
    checks that. The rest check the small evidence helpers on tiny made-up data.
"""

import re
from pathlib import Path

import pandas as pd

from src.answers_evidence import (
    subPrintConsequencesEvidence, subPrintGraphEvidence, subPrintTrainingEvidence, tplIngestCounts,
)


def strPlain(strText: str) -> str:
    """Remove the bold and underline codes, so a test can read the text."""
    return re.sub(r"\x1b\[[0-9;]*m", "", strText)


def test_stage_files_hold_no_assessment_steps():
    """The stage files only run the pipeline. Every subPrintStep call lives in answers_evidence.py."""
    pthSrc = Path(__file__).resolve().parent.parent / "src"
    for strStage in ["ingest", "validate", "train", "evaluate", "compare", "predict", "plots"]:
        assert "subPrintStep(" not in (pthSrc / f"{strStage}.py").read_text(encoding="utf-8"), strStage
    assert "subPrintStep(" in (pthSrc / "answers_evidence.py").read_text(encoding="utf-8")


def test_ingest_counts_follow_the_cleaning_steps(dctConfig):
    """One duplicate and one row below the minimum are removed, and each count is reported."""
    dctConfig["data"]["value_column"] = "feature_a"
    dctConfig["data"]["keep_columns"] = ["feature_a", "feature_b", "target"]
    dctConfig["data"]["min_value"] = 1
    dfRaw = pd.DataFrame({
        "feature_a": [0, 1, 2, 3, 3],
        "feature_b": [0, 1, 2, 0, 0],
        "target": [0, 1, 0, 1, 1],
    })
    assert tplIngestCounts(dctConfig, dfRaw) == (5, 4, 4, 3)


def test_consequences_evidence_prints_the_error_figures(capsys):
    """The false alarm share is the false positives divided by everything flagged."""
    dfCounts = pd.DataFrame([{"true_positive": 159, "false_negative": 99,
                              "false_positive": 552, "true_negative": 1030}])
    subPrintConsequencesEvidence(dfCounts)
    strOut = strPlain(capsys.readouterr().out)
    assert "99 of 258" in strOut
    assert "77.6%" in strOut


def test_training_evidence_says_so_when_there_is_no_model_card(dctConfig, tmp_path, capsys):
    """A missing model card gives a clear message, not a crash."""
    dctConfig["approval"]["model_card_path"] = str(tmp_path / "missing_card.json")
    subPrintTrainingEvidence(dctConfig)
    assert "Run python -m src.train first" in strPlain(capsys.readouterr().out)


def test_graph_evidence_says_so_when_there_are_no_graphs(dctConfig, tmp_path, capsys):
    """An empty or missing graph folder gives a clear message, not a crash."""
    dctConfig["plots"]["folder"] = str(tmp_path / "no_figures")
    subPrintGraphEvidence(dctConfig)
    assert "Run python -m src.plots first" in strPlain(capsys.readouterr().out)


def test_model_score_evidence_prints_scores_and_gap(dctConfig, tmp_path, capsys):
    import pandas as pd
    from src.answers_evidence import subPrintModelScoreEvidence
    pd.DataFrame({
        "model": ["baseline", "logistic_regression", "random_forest"],
        "role": ["baseline", "candidate", "candidate"],
        "cv_auc_mean": [0.5, 0.606, 0.599], "cv_auc_std": [0.0, 0.020, 0.024], "roc_auc": [0.5, 0.639, 0.636],
    }).to_csv(tmp_path / "t.csv", index=False)
    dctConfig["outputs"]["comparison_path"] = str(tmp_path / "t.csv")
    dctConfig["comparison"]["cv_folds"] = 5
    subPrintModelScoreEvidence(dctConfig)
    strOut = capsys.readouterr().out
    assert "test AUC 0.639" in strOut and "cross-validated AUC 0.6060" in strOut
    assert "a gap of 0.0070" in strOut and "smaller than the spread" in strOut


def test_model_score_evidence_without_table(dctConfig, tmp_path, capsys):
    from src.answers_evidence import subPrintModelScoreEvidence
    dctConfig["outputs"]["comparison_path"] = str(tmp_path / "missing.csv")
    subPrintModelScoreEvidence(dctConfig)
    assert "Run python -m src.compare first" in capsys.readouterr().out


def test_diagram_evidence_without_png(tmp_path, monkeypatch, capsys):
    from src import answers_evidence
    monkeypatch.setattr(answers_evidence, "STR_DIAGRAM_FILE", str(tmp_path / "none.png"))
    answers_evidence.subPrintDiagramEvidence()
    assert "Export the slide as a JPEG" in capsys.readouterr().out
