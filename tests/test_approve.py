"""Tests for the human sign-off on the saved model."""
import json

import pytest

from src.approve import ModelNotApprovedError, dctRecordSignoff, subRequireSignoff
from src.tracking import strFileFingerprint


@pytest.fixture
def dctSetup(dctConfig, tmp_path):
    """A fake saved model and a card that passed the gate, with sign-off required."""
    pthModel = tmp_path / "model.joblib"
    pthModel.write_bytes(b"model one")
    pthCard = tmp_path / "card.json"
    pthCard.write_text(json.dumps({
        "model_fingerprint": strFileFingerprint(pthModel), "gate": {"passed": True}}), encoding="utf-8")
    dctConfig["model"]["path"] = str(pthModel)
    dctConfig["approval"]["model_card_path"] = str(pthCard)
    dctConfig["approval"]["require_signoff"] = True
    dctConfig["approval"]["signoff_path"] = str(tmp_path / "signoff.json")
    return dctConfig, pthModel, pthCard


def test_unsigned_model_is_refused(dctSetup):
    with pytest.raises(ModelNotApprovedError, match="no sign-off"):
        subRequireSignoff(dctSetup[0])


def test_signed_model_is_allowed(dctSetup):
    dctConfig = dctSetup[0]
    dctRecordSignoff(dctConfig, "Dr Smith", "Clinical owner", "approved", "Reviewed the evidence.")
    subRequireSignoff(dctConfig)


def test_switch_off_skips_the_check(dctSetup):
    dctSetup[0]["approval"]["require_signoff"] = False
    subRequireSignoff(dctSetup[0])


def test_rejected_model_is_refused(dctSetup):
    dctRecordSignoff(dctSetup[0], "Dr Smith", "Clinical owner", "rejected", "Too weak.")
    with pytest.raises(ModelNotApprovedError, match="rejected"):
        subRequireSignoff(dctSetup[0])


def test_changed_model_needs_new_signoff(dctSetup):
    dctConfig, pthModel, _ = dctSetup
    dctRecordSignoff(dctConfig, "Dr Smith", "Clinical owner", "approved", "Reviewed the evidence.")
    pthModel.write_bytes(b"model two")
    with pytest.raises(ModelNotApprovedError, match="changed"):
        subRequireSignoff(dctConfig)


def test_cannot_approve_a_model_that_failed_the_gate(dctSetup):
    dctConfig, _, pthCard = dctSetup
    dctCard = json.loads(pthCard.read_text(encoding="utf-8"))
    dctCard["gate"]["passed"] = False
    pthCard.write_text(json.dumps(dctCard), encoding="utf-8")
    with pytest.raises(ModelNotApprovedError, match="failed"):
        dctRecordSignoff(dctConfig, "Dr Smith", "Clinical owner", "approved", "Reviewed the evidence.")


def test_cannot_sign_off_a_model_the_card_does_not_describe(dctSetup):
    dctConfig, pthModel, _ = dctSetup
    pthModel.write_bytes(b"model two")
    with pytest.raises(ModelNotApprovedError, match="card describes"):
        dctRecordSignoff(dctConfig, "Dr Smith", "Clinical owner", "approved", "Reviewed the evidence.")


def test_name_role_and_reason_are_required(dctSetup):
    with pytest.raises(ValueError):
        dctRecordSignoff(dctSetup[0], " ", "Clinical owner", "approved", "Fine.")
    with pytest.raises(ValueError):
        dctRecordSignoff(dctSetup[0], "Dr Smith", "Clinical owner", "maybe", "Fine.")


def test_not_approved_message_gives_the_command(dctSetup):
    from src.approve import strNotApprovedMessage
    dctConfig = dctSetup[0]
    dctConfig["approval"]["evidence_path"] = "reports/evaluation_evidence.md"
    strMessage = strNotApprovedMessage(ModelNotApprovedError("This model has no sign-off."), dctConfig)
    assert "MODEL NOT APPROVED" in strMessage
    assert 'python -m src.approve --name "..." --role "..." --decision approved --reason "..."' in strMessage


def test_exit_not_approved_stops_with_code_1(dctSetup, capsys):
    from src.approve import subExitNotApproved
    dctSetup[0]["approval"]["evidence_path"] = "reports/evaluation_evidence.md"
    with pytest.raises(SystemExit) as objExit:
        subExitNotApproved(ModelNotApprovedError("x"), dctSetup[0])
    assert objExit.value.code == 1
    assert "MODEL NOT APPROVED" in capsys.readouterr().err
