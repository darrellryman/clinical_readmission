"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    conftest.py
    
    Shared test setup. pytest loads this file automatically, and any fixture defined
    here can be used by name in every test file, with no import needed.
    
    The fixtures are named dctConfig and dfSample. pytest matches a fixture to a
    test by the parameter name, so a test that wants the config writes dctConfig as
    a parameter.
"""

import pandas as pd
import pytest


@pytest.fixture
def dctConfig() -> dict:
    """A config dictionary for tests to share.

    It has the same shape as config.yaml but is written out here, so the tests
    keep passing even if someone later edits the real config.yaml (for example
    to point at their own columns). pytest builds a fresh copy for each test,
    so a test can change it freely without affecting any other test.
    """
    return {
        "project": {"random_state": 42, "client_question": "Test question?"},
        "data": {
            "source": "file",
            "raw_path": "data/raw/data.csv",
            "raw_url": None,
            "download_timeout": 30,
            "download_retries": 3,
            "processed_path": "data/processed/clean.csv",
            "keep_columns": ["feature_a", "feature_b", "target"],
            "allowed_values": {},
            "min_value": None,
            "max_value": None,
            "value_column": "age",
            "new_data_fraction": 0.0,
            "new_data_path": "data/new/new_data.csv",
            "new_data_answers_path": "data/new/new_data_answers.csv",
        },
        "validation": {"min_rows": 10, "column_ranges": {}},
        "features": {
            "feature_columns": ["feature_a", "feature_b"],
            "target_column": "target",
            "test_size": 0.2,
            "stratify": True,
            "categorical_columns": {},
            "difference_columns": {},
            "excluded_columns": [],
        },
        "model": {"path": "artifacts/model.joblib", "max_iter": 1000, "C": 1.0, "class_weight": None},
        "evaluation": {
            "average": "weighted", "positive_label": 1,
            "group_columns": [], "binned_columns": {},
        },
        "outputs": {
            "metrics_path": "reports/metrics.json",
            "predictions_path": "reports/predictions.csv",
            "prediction_column": "prediction",
            "rtf_path": "reports/{stage}_output.rtf",
            "comparison_path": "reports/model_comparison.csv",
            "features_path": "data/processed/features.csv",
        },
        "tracking": {"enabled": False, "uri": "sqlite:///mlflow_test.db", "experiment_name": "test"},
        "comparison": {
            "cv_folds": 3,
            "baseline": {"type": "dummy", "strategy": "most_frequent"},
            "benchmarks": {"rule_a": {"type": "rule", "column": "feature_a", "threshold": 10}},
            "candidates": {
                "logistic_regression": {"type": "logistic", "max_iter": 1000, "C": 1.0, "class_weight": None},
                "random_forest": {
                    "type": "random_forest", "n_estimators": 20, "max_depth": 3,
                    "min_samples_leaf": 2, "class_weight": None,
                },
            },
        },
        "approval": {
            "enabled": True,
            "min_auc_gain": 0.05,
            "model_card_path": "artifacts/model_card.json",
            "evidence_path": "reports/evaluation_evidence.md",
            "intended_use": "Test use.",
            "not_for": "Anything real.",
            "known_limits": ["Test data only."],
            "recommendation": "Do not use.",
        },
        "plots": {
            "enabled": True, "folder": "reports/figures", "dpi": 60,
            "palette": ["#888888", "#E0A100", "#0F8B8D", "#12263A"],
            "group_metrics": ["recall", "precision"],
        },
        "service": {"host": "127.0.0.1", "port": 8000, "max_patients": 5},
        "testing": {"rtf_path": "reports/test/{stage}_output.rtf"},
    }


@pytest.fixture
def dfSample() -> pd.DataFrame:
    """A small, valid dataset for tests to share.

    Twenty rows, two numeric features and a balanced binary target (ten 0s and
    ten 1s). It matches the columns in the dctConfig fixture above, so it passes
    validation.
    """
    return pd.DataFrame(
        {
            "feature_a": list(range(20)),
            "feature_b": [intRow % 3 for intRow in range(20)],  # Repeats 0, 1, 2, 0, 1, 2 ...
            "target": [0] * 10 + [1] * 10,
        }
    )
