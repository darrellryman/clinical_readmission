"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    test_display.py
    
    Tests for src/display.py: the dataset profile prints the right facts and
    handles a missing dataset without crashing, and the RTF helpers save printed
    output to a file.
"""

import pandas as pd
import pytest

from src.display import (
    bDisplayDatasetDetails,
    objCaptureToRtf,
    pthResolveRtfPath,
    pthSaveRtf,
    subPrintStep,
)


def test_profile_returns_true_and_prints_the_counts(dfSample, capsys):
    """A real DataFrame is profiled: it returns True and the output has the row and column counts."""
    assert bDisplayDatasetDetails(dfSample) is True
    strOutput = capsys.readouterr().out
    # dfSample has 20 rows and 3 columns, so 60 cells.
    assert "Rows:" in strOutput and "20" in strOutput
    assert "Cells:" in strOutput and "60" in strOutput


def test_profile_reports_missing_and_duplicates(capsys):
    """One empty cell and one repeated row both show up in the printout."""
    dfData = pd.DataFrame({"a": [1, 1, None], "b": [2, 2, 3]})
    bDisplayDatasetDetails(dfData)
    strOutput = capsys.readouterr().out
    assert "Total missing values in the whole dataset" in strOutput
    assert "Number of duplicated Rows" in strOutput


def test_profile_handles_none(capsys):
    """None is reported as an error and returns False instead of raising."""
    assert bDisplayDatasetDetails(None) is False
    assert "Dataset is NULL" in capsys.readouterr().out


def test_resolve_rtf_path_fills_in_the_stage_name():
    """The text {stage} in the configured name becomes the stage name."""
    assert str(pthResolveRtfPath("reports/{stage}_output.rtf", "ingest")).replace("\\", "/") == "reports/ingest_output.rtf"


def test_save_rtf_writes_a_valid_rtf_file(tmp_path):
    """The file starts with the RTF marker, ends with a closing brace, and the folder is created."""
    pthFile = pthSaveRtf("hello\nworld", tmp_path / "new_folder" / "out.rtf")
    strRtf = pthFile.read_text(encoding="ascii")
    assert strRtf.startswith("{\\rtf1")
    assert strRtf.endswith("}")
    assert "hello\\par" in strRtf and "world" in strRtf


def test_save_rtf_escapes_rtf_commands_and_removes_terminal_codes(tmp_path):
    """Braces and backslashes can't be read as RTF commands, and bold codes are removed."""
    pthFile = pthSaveRtf("\033[1mbold\033[0m {x} a\\b", tmp_path / "out.rtf")
    strRtf = pthFile.read_text(encoding="ascii")
    assert "\x1b" not in strRtf
    assert "\\{x\\}" in strRtf
    assert "a\\\\b" in strRtf


def test_save_rtf_keeps_non_ascii_characters(tmp_path):
    """A character such as an accented e is written as an RTF code, not lost or broken."""
    pthFile = pthSaveRtf("caf\u00e9", tmp_path / "out.rtf")
    assert "\\u233?" in pthFile.read_text(encoding="ascii")


def test_capture_sends_output_to_the_file_not_the_screen(tmp_path, capsys):
    """Printed text lands in the RTF file. The screen only gets the 'saved to' line."""
    strTemplate = str(tmp_path / "{stage}_out.rtf")
    with objCaptureToRtf(strTemplate, "demo") as pthFile:
        print("secret line")
    strScreen = capsys.readouterr().out
    assert "secret line" not in strScreen
    assert "Output saved to" in strScreen
    assert pthFile.name == "demo_out.rtf"
    assert "secret line" in pthFile.read_text(encoding="ascii")


def test_capture_saves_the_file_even_when_the_block_fails(tmp_path):
    """If the code inside stops with an error, what printed before it is still saved."""
    strTemplate = str(tmp_path / "{stage}_out.rtf")
    with pytest.raises(ValueError):
        with objCaptureToRtf(strTemplate, "demo"):
            print("before the crash")
            raise ValueError("boom")
    assert "before the crash" in (tmp_path / "demo_out.rtf").read_text(encoding="ascii")


def test_blank_rtf_path_prints_to_the_screen(capsys):
    """A blank path means no capture, so the output goes to the screen as usual."""
    with objCaptureToRtf("", "demo") as pthFile:
        print("on screen")
    assert pthFile is None
    assert "on screen" in capsys.readouterr().out


def test_step_prints_question_and_placeholder_when_no_answer(capsys):
    """A step with no answer says so, so a missing answer is easy to spot."""
    subPrintStep("Step 1 - Define the problem.")
    strOut = capsys.readouterr().out
    assert "Step 1 - Define the problem." in strOut
    assert "(answer not written yet)" in strOut


def test_step_prints_the_answer_when_given(capsys):
    """A written answer replaces the placeholder."""
    subPrintStep("Step 1 - Define the problem.", "Predict 30-day readmission.")
    strOut = capsys.readouterr().out
    assert "Predict 30-day readmission." in strOut
    assert "not written yet" not in strOut


def test_step_text_is_bold_and_italic_in_the_rtf_file(tmp_path, capsys):
    """Saved to RTF, the question is italic, "Answer:" is bold italic, and no terminal codes remain."""
    subPrintStep("Step 1 - Define the problem.", "An answer.")
    pthFile = pthSaveRtf(capsys.readouterr().out, tmp_path / "step.rtf")
    strRtf = pthFile.read_text()
    assert "\\i Step 1 - Define the problem." in strRtf
    assert "\\b \\i Answer:" in strRtf
    assert "\\b0\\i0" in strRtf
    assert "\x1b" not in strRtf


def test_underline_and_bold_headings_keep_their_style_in_rtf(tmp_path):
    """The existing bold and underline headings now show as bold and underline in the RTF."""
    pthFile = pthSaveRtf("\033[1m\033[4mHeading\033[0m", tmp_path / "h.rtf")
    strRtf = pthFile.read_text()
    assert "\\b \\ul Heading" in strRtf


def test_a_locked_rtf_file_prints_a_warning_and_does_not_crash(tmp_path, capsys, monkeypatch):
    """If the RTF cannot be written because it is open elsewhere, the screen says so."""
    import src.display as modDisplay

    def subLocked(strText, varPath):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(modDisplay, "pthSaveRtf", subLocked)
    with objCaptureToRtf(str(tmp_path / "{stage}.rtf"), "demo"):
        print("hello")
    strScreen = capsys.readouterr().out
    assert "was NOT updated" in strScreen
    assert "demo.rtf" in strScreen
    assert "Output saved" not in strScreen


def test_the_rtf_file_name_is_lower_case_whatever_the_stage_label(tmp_path):
    """A stage labelled Ingest still writes ingest_output.rtf, so DVC and CI find it on any system."""
    from src.display import pthResolveRtfPath
    assert pthResolveRtfPath("reports/{stage}_output.rtf", "Ingest").name == "ingest_output.rtf"
