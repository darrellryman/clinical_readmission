"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    approve.py

    Human sign-off for the saved model.

    Passing the automated gate in train.py only shows the model beats guessing.
    Before the model scores anyone, a named person has to read the evidence and
    record a decision. This file does that in three small steps:

        1. dctRecordSignoff()   writes the person's name, role, date, decision and
                                reason to the sign-off file (approval.signoff_path),
                                with the fingerprint of the model they approved.
        2. subRequireSignoff()  is called by predict.py and serve.py before they load
                                the model. It stops them if there is no approval, or
                                if the model file is not the one that was approved.
        3. A retrained model has a new fingerprint, so the old sign-off no longer
                                matches and the new model needs its own.

    The switch is approval.require_signoff in config.yaml.

    The sign-off is its own file, not part of the model card, because train.py and
    DVC own the card and rewrite it on every run. Commit the sign-off file to Git.
    The history then shows who approved what and when. It is a plain JSON file, so
    someone could edit it by hand. In a real system it would be signed, or kept in
    a separate system with its own access control.

    Run from the project root, after python -m src.train:
        python -m src.approve --name "Dr Jane Smith" --role "Clinical owner" \\
            --decision approved --reason "Reviewed evidence. Pilot in shadow mode only."
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

from src.config import subAddConfigArgument, dctLoadConfig
from src.tracking import strFileFingerprint

# The two decisions a reviewer can record.
LST_DECISIONS = ["approved", "rejected"]


class ModelNotApprovedError(Exception):
    """Raised when a model is used without a valid sign-off from a named person."""


def dctReadCard(varPath) -> dict:
    """Read the model card from its JSON file.

    Args:
        varPath: The card's path, from approval.model_card_path.

    Returns:
        The card as a dictionary.

    Raises:
        ModelNotApprovedError: If there is no card. Run src.train first.
    """
    pthFile = Path(varPath)
    if not pthFile.is_file():
        raise ModelNotApprovedError(f"No model card at {pthFile}. Run src.train first.")
    return json.loads(pthFile.read_text(encoding="utf-8"))


STR_DEFAULT_SIGNOFF = "approvals/signoff.json"


def pthSignoffFile(dctConfig: dict) -> Path:
    """Return the path of the sign-off file (approval.signoff_path, with a default)."""
    return Path(dctConfig["approval"].get("signoff_path", STR_DEFAULT_SIGNOFF))


def dctRecordSignoff(dctConfig: dict, strName: str, strRole: str, strDecision: str, strReason: str,
                     strModelPath: str = None) -> dict:
    """Write a person's decision to the sign-off file.

    A model that failed the automated gate cannot be approved. The card must also
    describe the model file that is on disk now, so nobody signs off on a model
    that has since been replaced.

    Args:
        dctConfig: The loaded config.yaml settings. Uses approval.model_card_path,
            approval.signoff_path and model.path.
        strName: Who is deciding. Required.
        strRole: Their role, for example "Clinical owner". Required.
        strDecision: "approved" or "rejected".
        strReason: Why. Required, so the record says more than yes or no.
        strModelPath: The model file. Default: model.path.

    Returns:
        The sign-off record that was written.

    Raises:
        ValueError: If a name, role or reason is blank, or the decision is not one of the two.
        ModelNotApprovedError: If the card is missing, the model failed the gate, or the
            card does not match the model file.
    """
    if strDecision not in LST_DECISIONS:
        raise ValueError(f"Decision must be one of {LST_DECISIONS}, not '{strDecision}'.")
    if not (strName.strip() and strRole.strip() and strReason.strip()):
        raise ValueError("A name, a role and a reason are all required.")

    dctCard = dctReadCard(dctConfig["approval"]["model_card_path"])
    strFingerprint = strFileFingerprint(strModelPath or dctConfig["model"]["path"])

    if strDecision == "approved" and not dctCard["gate"]["passed"]:
        raise ModelNotApprovedError("The model failed the automated gate, so it cannot be approved.")
    if dctCard["model_fingerprint"] != strFingerprint:
        raise ModelNotApprovedError(
            f"The card describes model {dctCard['model_fingerprint']} but the file is {strFingerprint}. "
            "Run src.train again, then review the new model.")

    dctSignoff = {
        "decision": strDecision,
        "name": strName.strip(),
        "role": strRole.strip(),
        "reason": strReason.strip(),
        "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "model_fingerprint": strFingerprint,
    }
    pthSignoff = pthSignoffFile(dctConfig)
    pthSignoff.parent.mkdir(parents=True, exist_ok=True)
    pthSignoff.write_text(json.dumps(dctSignoff, indent=2), encoding="utf-8")
    return dctSignoff


def subRequireSignoff(dctConfig: dict, strModelPath: str = None) -> None:
    """Stop unless a named person has approved the model file about to be used.

    Call it before loading a model to score patients. It does nothing when
    approval.require_signoff is false or missing.

    Args:
        dctConfig: The loaded config.yaml settings. Uses approval.require_signoff,
            approval.signoff_path and model.path.
        strModelPath: The model file. Default: model.path.

    Returns:
        None. It only raises when something is wrong.

    Raises:
        ModelNotApprovedError: If there is no sign-off, the decision is not
            "approved", or the model file is not the one that was approved.
    """
    if not dctConfig["approval"].get("require_signoff", False):
        return

    pthSignoff = pthSignoffFile(dctConfig)
    dctSignoff = json.loads(pthSignoff.read_text(encoding="utf-8")) if pthSignoff.is_file() else None
    if not dctSignoff:
        raise ModelNotApprovedError("This model has no sign-off.")
    if dctSignoff.get("decision") != "approved":
        raise ModelNotApprovedError(f"This model was {dctSignoff.get('decision')} by {dctSignoff.get('name')}.")

    strFingerprint = strFileFingerprint(strModelPath or dctConfig["model"]["path"])
    if dctSignoff.get("model_fingerprint") != strFingerprint:
        raise ModelNotApprovedError("The model file has changed since it was approved. It needs a new sign-off.")


def strNotApprovedMessage(objError: Exception, dctConfig: dict) -> str:
    """Build the message shown when a model is blocked for lack of approval.

    Args:
        objError: The ModelNotApprovedError that was raised. Its text says why.
        dctConfig: The loaded config.yaml settings. Uses approval.model_card_path
            and approval.signoff_path.

    Returns:
        Plain text that says what is wrong and the exact command that fixes it.
    """
    return (
        "\nMODEL NOT APPROVED\n"
        f"{objError}\n\n"
        "Approval from a named person must be obtained before this model can be used.\n"
        f"Review {dctConfig['approval']['model_card_path']} and {dctConfig['approval']['evidence_path']}, then run:\n\n"
        '    python -m src.approve --name "..." --role "..." --decision approved --reason "..."\n\n'
        f"The decision is recorded in {pthSignoffFile(dctConfig)}. Commit that file to Git.\n"
    )


def subExitNotApproved(objError: Exception, dctConfig: dict) -> None:
    """Show the not-approved message on the screen and stop with exit code 1.

    Args:
        objError: The ModelNotApprovedError that was raised.
        dctConfig: The loaded config.yaml settings.

    Returns:
        None. It never returns. It ends the program, so a script or CI job sees a failure.
    """
    print(strNotApprovedMessage(objError, dctConfig), file=sys.stderr)
    sys.exit(1)


def subMain() -> None:
    """Command line entry point: record a person's decision about the saved model."""
    objParser = argparse.ArgumentParser(description="Record a named person's decision about the saved model.")
    subAddConfigArgument(objParser)
    objParser.add_argument("--name", required=True, help="Who is deciding")
    objParser.add_argument("--role", required=True, help="Their role, for example 'Clinical owner'")
    objParser.add_argument("--decision", required=True, choices=LST_DECISIONS)
    objParser.add_argument("--reason", required=True, help="Why")
    objArgs = objParser.parse_args()

    dctConfig = dctLoadConfig(objArgs.config)
    dctCard = dctRecordSignoff(dctConfig, objArgs.name, objArgs.role, objArgs.decision, objArgs.reason)
    print(f"{dctCard['decision'].capitalize()} by {dctCard['name']} ({dctCard['role']}). "
          f"Recorded in {pthSignoffFile(dctConfig)}. Commit that file to Git.")


if __name__ == "__main__":
    subMain()
