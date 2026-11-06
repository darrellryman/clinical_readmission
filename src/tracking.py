"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    tracking.py

    Shared experiment tracking for the pipeline, using MLflow.

    Each stage that makes something worth recording (train, evaluate, compare)
    calls subLogRun() at the end. MLflow then keeps one run per call with its
    settings (parameters), scores (metrics), files (artefacts) and tags. Runs
    pile up, so you can see how the results changed between changes to config.yaml.

    Every run is tagged with a fingerprint of the data it used, the Git commit
    of the code, and the stage name. With those tags, another person can work
    out which data, code and configuration produced a model.

    Tracking is optional. If tracking.enabled in config.yaml is false, or the
    mlflow package isn't installed, nothing is recorded and the stage carries
    on normally. That matters because mlflow is one of the harder packages to
    install on Windows on ARM.

    The tracking location and experiment name come from config.yaml (the
    tracking section).
"""

# hashlib makes a short fingerprint of a file, so two runs on the same data
# show the same fingerprint and a changed file shows a different one.
import hashlib

# subprocess asks Git which commit the code is on.
import subprocess
from pathlib import Path


def strFileFingerprint(varPath) -> str:
    """Return a short fingerprint of a file's contents.

    Args:
        varPath: The file to fingerprint, as text or a Path.

    Returns:
        The first 12 characters of the file's SHA-256 hash, or "missing" if the
        file does not exist. Two files with the same contents always give the
        same fingerprint.
    """
    pthFile = Path(varPath)
    if not pthFile.is_file():
        return "missing"
    objHash = hashlib.sha256()
    # Read in chunks so a large file never has to fit in memory at once.
    with pthFile.open("rb") as objHandle:
        for bytChunk in iter(lambda: objHandle.read(1024 * 1024), b""):
            objHash.update(bytChunk)
    return objHash.hexdigest()[:12]


def strGitCommit() -> str:
    """Return the short Git commit of the code, or "unknown" if Git can't say.

    Returns:
        The short commit hash, with "-dirty" added when there are uncommitted
        changes (so you know the run did not use exactly that commit). If this
        is not a Git repository, or Git is not installed, returns "unknown".
    """
    try:
        strCommit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        strStatus = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
        return strCommit + ("-dirty" if strStatus else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def bLogRun(dctConfig: dict, strStage: str, strRunName: str, dctParams: dict = None,
            dctMetrics: dict = None, lstArtifacts: list = None, dctTags: dict = None) -> bool:
    """Record one run in MLflow, if tracking is switched on and available.

    Args:
        dctConfig: The loaded config.yaml settings. Uses tracking.enabled,
            tracking.uri, tracking.experiment_name and data.processed_path.
        strStage: The stage making the record, for example "train". Saved as a tag.
        strRunName: A label for the run in the MLflow list, for example
            "logistic_regression".
        dctParams: Settings to record, such as the model settings.
        dctMetrics: Scores to record. Values must be numbers.
        lstArtifacts: Paths of files to attach to the run, such as the model.
            Paths that don't exist are skipped.
        dctTags: Extra labels to attach.

    Returns:
        True if a run was recorded. False if tracking is off, mlflow isn't
        installed, or the recording failed. The reason is printed, and the
        stage carries on either way.
    """
    dctTracking = dctConfig["tracking"]
    if not dctTracking["enabled"]:
        print("MLflow tracking is switched off in config.yaml, so this run was not recorded.")
        return False

    # Import here, not at the top of the file, so the pipeline still runs when
    # mlflow isn't installed.
    try:
        import mlflow
    except ImportError:
        print("MLflow is not installed, so this run was not recorded. Install it with: pip install mlflow")
        return False

    try:
        mlflow.set_tracking_uri(dctTracking["uri"])
        mlflow.set_experiment(dctTracking["experiment_name"])
        with mlflow.start_run(run_name=strRunName) as objRun:
            # Settings. MLflow stores parameters as text, so convert each value.
            for strKey, varValue in (dctParams or {}).items():
                mlflow.log_param(strKey, str(varValue)[:500])

            # Scores. NaN can't be recorded, so those are skipped.
            for strKey, varValue in (dctMetrics or {}).items():
                if varValue == varValue:
                    mlflow.log_metric(strKey, float(varValue))

            # Tags that tie the run to the data, the code and the stage.
            dctAllTags = {
                "stage": strStage,
                "git_commit": strGitCommit(),
                "data_fingerprint": strFileFingerprint(dctConfig["data"]["processed_path"]),
                "raw_data_fingerprint": strFileFingerprint(dctConfig["data"]["raw_path"]),
                "config_fingerprint": strFileFingerprint("config.yaml"),
            }
            dctAllTags.update(dctTags or {})
            mlflow.set_tags(dctAllTags)

            # Files. The config is always attached so the settings travel with the run.
            for varPath in list(lstArtifacts or []) + ["config.yaml"]:
                if Path(varPath).is_file():
                    mlflow.log_artifact(str(varPath))

            print(f"Run recorded in MLflow: experiment '{dctTracking['experiment_name']}', "
                  f"run '{strRunName}', id {objRun.info.run_id}")
        return True
    except Exception as objError:  # noqa: BLE001 - tracking must never stop the pipeline
        print(f"MLflow could not record this run ({type(objError).__name__}: {objError}). The stage carried on.")
        return False
