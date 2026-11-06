"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    test_ingest.py
    
    Tests for src/ingest.py: loading a CSV, keeping only the configured columns,
    and saving the result.
"""

import pandas as pd
import pytest

from src.ingest import dfCleanData, dfFilterAllowedValues, dfFilterValueRange, dfKeepColumns, dfLoadCsv, pthSaveCsv


def test_keep_columns_keeps_only_the_listed_columns_in_order(dfSample):
    """The result has exactly the requested columns, in the requested order."""
    dfOut = dfKeepColumns(dfSample, ["target", "feature_a"])
    assert list(dfOut.columns) == ["target", "feature_a"]
    assert len(dfOut) == len(dfSample)


def test_keep_columns_does_not_change_the_original(dfSample):
    """Dropping columns from the copy leaves the original table whole."""
    dfKeepColumns(dfSample, ["feature_a"])
    assert list(dfSample.columns) == ["feature_a", "feature_b", "target"]


def test_keep_columns_names_every_missing_column(dfSample):
    """A typo in the config is reported by name, and all missing names are listed."""
    with pytest.raises(KeyError, match="nope.*also_nope"):
        dfKeepColumns(dfSample, ["feature_a", "nope", "also_nope"])


def test_clean_removes_exact_duplicates_and_keeps_the_first():
    """A repeated row is removed once, and the rows are renumbered from 0."""
    dfData = pd.DataFrame({"a": [1, 2, 2, 3], "b": ["x", "y", "y", "z"]})
    dfOut = dfCleanData(dfData)
    assert len(dfOut) == 3
    assert list(dfOut.index) == [0, 1, 2]


def test_clean_tidies_headers_and_text():
    """Headers lose spaces and capitals, and text values lose surrounding spaces."""
    dfData = pd.DataFrame({" Age ": [1, 2], "Smoker": ["yes ", " no"]})
    dfOut = dfCleanData(dfData)
    assert list(dfOut.columns) == ["age", "smoker"]
    assert list(dfOut["smoker"]) == ["yes", "no"]


def test_clean_treats_spaced_text_as_a_duplicate():
    """'yes ' and 'yes' are the same value, so the rows collapse into one."""
    dfData = pd.DataFrame({"smoker": ["yes ", "yes"]})
    assert len(dfCleanData(dfData)) == 1


def test_clean_leaves_the_original_untouched():
    """Cleaning works on a copy, so the caller's table keeps its duplicates."""
    dfData = pd.DataFrame({"a": [1, 1]})
    dfCleanData(dfData)
    assert len(dfData) == 2


def test_filter_keeps_only_allowed_values():
    """Rows with a value off the list are removed, and the rest are renumbered."""
    dfData = pd.DataFrame({"sex": ["Female", "Other", "Male", "Female"], "n": [1, 2, 3, 4]})
    dfOut = dfFilterAllowedValues(dfData, {"sex": ["Female", "Male"]})
    assert list(dfOut["sex"].value_counts().index.sort_values()) == ["Female", "Male"]
    assert len(dfOut) == 3
    assert list(dfOut.index) == [0, 1, 2]


def test_filter_removes_blank_values():
    """A missing value is not on any list, so that row is removed too."""
    dfData = pd.DataFrame({"sex": ["Female", None, "Male"]})
    assert len(dfFilterAllowedValues(dfData, {"sex": ["Female", "Male"]})) == 2


def test_filter_with_empty_rules_keeps_every_row():
    """An empty dictionary means no rules, so nothing is removed."""
    dfData = pd.DataFrame({"sex": ["Female", "Other"]})
    assert len(dfFilterAllowedValues(dfData, {})) == 2


def test_filter_names_a_missing_column():
    """A rule for a column that isn't in the data is reported by name."""
    with pytest.raises(KeyError, match="colour"):
        dfFilterAllowedValues(pd.DataFrame({"sex": ["Male"]}), {"colour": ["red"]})


def test_filter_leaves_the_original_untouched():
    """Filtering works on a copy, so the caller's table keeps all its rows."""
    dfData = pd.DataFrame({"sex": ["Female", "Other"]})
    dfFilterAllowedValues(dfData, {"sex": ["Female"]})
    assert len(dfData) == 2


def test_save_then_load_round_trip(dfSample, tmp_path):
    """A saved CSV loads back with the same columns and rows."""
    pthFile = pthSaveCsv(dfSample, tmp_path / "out" / "clean.csv")
    dfBack = dfLoadCsv(pthFile)
    assert list(dfBack.columns) == list(dfSample.columns)
    assert len(dfBack) == len(dfSample)


def test_load_csv_missing_file(tmp_path):
    """Loading a file that doesn't exist raises a clear error."""
    with pytest.raises(FileNotFoundError):
        dfLoadCsv(tmp_path / "nope.csv")


def dfPatients(intRows: int = 200) -> pd.DataFrame:
    """Test helper: a small table with a rare target (about 20 percent positive)."""
    return pd.DataFrame({
        "feature_a": range(intRows),
        "feature_b": [x % 7 for x in range(intRows)],
        "target": [1 if x % 5 == 0 else 0 for x in range(intRows)],
    })


def test_holdout_sets_aside_the_configured_share(dctConfig):
    """With a fraction of 0.1, about 10 percent of rows go to the new patients."""
    from src.ingest import tplHoldOutNewPatients
    dctConfig["data"]["new_data_fraction"] = 0.1
    dfKept, dfNew = tplHoldOutNewPatients(dfPatients(), dctConfig)
    assert len(dfNew) == 20
    assert len(dfKept) == 180


def test_holdout_keeps_the_target_mix_in_both_groups(dctConfig):
    """The readmission rate is the same in the set-aside rows as in the rest."""
    from src.ingest import tplHoldOutNewPatients
    dctConfig["data"]["new_data_fraction"] = 0.1
    dfKept, dfNew = tplHoldOutNewPatients(dfPatients(), dctConfig)
    assert dfKept["target"].mean() == pytest.approx(0.2, abs=0.01)
    assert dfNew["target"].mean() == pytest.approx(0.2, abs=0.01)


def test_holdout_is_repeatable_and_never_overlaps(dctConfig):
    """The same seed sets aside the same rows, and no row is in both groups."""
    from src.ingest import tplHoldOutNewPatients
    dctConfig["data"]["new_data_fraction"] = 0.1
    dfKept1, dfNew1 = tplHoldOutNewPatients(dfPatients(), dctConfig)
    dfKept2, dfNew2 = tplHoldOutNewPatients(dfPatients(), dctConfig)
    assert list(dfNew1["feature_a"]) == list(dfNew2["feature_a"])
    assert set(dfNew1["feature_a"]).isdisjoint(set(dfKept1["feature_a"]))


def test_fraction_zero_sets_nothing_aside(dctConfig):
    """A fraction of 0 keeps every row and returns an empty new-patients table."""
    from src.ingest import tplHoldOutNewPatients
    dfKept, dfNew = tplHoldOutNewPatients(dfPatients(), dctConfig)
    assert len(dfKept) == 200
    assert len(dfNew) == 0


def test_new_patients_files_keep_the_answer_out_of_the_questions(dctConfig, tmp_path):
    """The new-patients file has no target column, the answers file has only it, and rows line up."""
    from src.ingest import tplSaveNewPatients
    dctConfig["data"]["new_data_path"] = str(tmp_path / "new.csv")
    dctConfig["data"]["new_data_answers_path"] = str(tmp_path / "answers.csv")
    dfNew = dfPatients(10)
    pthNew, pthAnswers = tplSaveNewPatients(dfNew, dctConfig)
    dfQuestions = pd.read_csv(pthNew)
    dfAnswers = pd.read_csv(pthAnswers)
    assert "target" not in dfQuestions.columns
    assert list(dfAnswers.columns) == ["target"]
    assert list(dfAnswers["target"]) == list(dfNew["target"])
    assert len(dfQuestions) == len(dfAnswers)


def test_nothing_is_written_when_no_patients_are_set_aside(dctConfig):
    """An empty set-aside table writes no files and returns None for both paths."""
    from src.ingest import tplSaveNewPatients
    assert tplSaveNewPatients(dfPatients(0), dctConfig) == (None, None)


def test_value_range_removes_lower_values():
    """Patients below the minimum age are removed, the exact age is kept, rows are renumbered."""
    dfData = pd.DataFrame({"age": [0, 17, 18, 40], "n": [1, 2, 3, 4]})
    dfOut = dfFilterValueRange(dfData, "age", 18, None)
    assert list(dfOut["age"]) == [18, 40]
    assert list(dfOut.index) == [0, 1]


def test_value_range_removes_higher_values():
    """Patients above the maximum age are removed, and the exact maximum is kept."""
    dfData = pd.DataFrame({"age": [30, 99, 100, 101, 120]})
    assert list(dfFilterValueRange(dfData, "age", None, 100)["age"]) == [30, 99, 100]


def test_value_range_picks_out_one_band():
    """Setting both ends keeps one band, ends included, so adjacent bands do not overlap."""
    dfData = pd.DataFrame({"age": [10, 39, 40, 59, 60, 80]})
    assert list(dfFilterValueRange(dfData, "age", 40, 59)["age"]) == [40, 59]
    assert list(dfFilterValueRange(dfData, "age", 18, 39)["age"]) == [39]


def test_value_range_is_configurable():
    """A different minimum gives a different result, for countries with another age of majority."""
    dfData = pd.DataFrame({"age": [17, 18, 19, 20]})
    assert len(dfFilterValueRange(dfData, "age", 20, None)) == 1
    assert len(dfFilterValueRange(dfData, "age", 19, None)) == 2


def test_value_range_open_ends_keep_every_row():
    """Min None and max None turn the filter off, even when the age column is missing."""
    dfData = pd.DataFrame({"x": [1, 2]})
    assert len(dfFilterValueRange(dfData, "age", None, None)) == 2


def test_value_range_removes_blank_ages():
    """A patient with no age cannot be shown to be in the range, so the row is removed."""
    dfData = pd.DataFrame({"age": [30.0, None, 5.0]})
    assert len(dfFilterValueRange(dfData, "age", 18, None)) == 1
    assert len(dfFilterValueRange(dfData, "age", None, 50)) == 2


def test_value_range_names_a_missing_column():
    """A missing age column is reported by name."""
    with pytest.raises(KeyError, match="age"):
        dfFilterValueRange(pd.DataFrame({"x": [1]}), "age", 18, None)


def test_value_range_needs_numbers():
    """A text age column stops the run with a clear message."""
    with pytest.raises(TypeError, match="numbers"):
        dfFilterValueRange(pd.DataFrame({"age": ["old", "young"]}), "age", 18, 100)


def test_value_range_leaves_the_original_untouched():
    """Filtering works on a copy, so the caller's table keeps all its rows."""
    dfData = pd.DataFrame({"age": [5, 30]})
    dfFilterValueRange(dfData, "age", 18, 100)
    assert len(dfData) == 2


def test_value_range_works_on_another_column():
    """The filter is not tied to age: any numeric column can be the one that is filtered."""
    dfData = pd.DataFrame({"bmi": [15.0, 22.5, 31.0, 45.0], "age": [30, 30, 30, 30]})
    assert list(dfFilterValueRange(dfData, "bmi", 18.5, 40)["bmi"]) == [22.5, 31.0]


def test_value_range_allows_a_negative_minimum():
    """A column that can go below zero can be filtered, because 0 is no longer the open end."""
    dfData = pd.DataFrame({"change": [-5, -1, 0, 3]})
    assert list(dfFilterValueRange(dfData, "change", -2, None)["change"]) == [-1, 0, 3]


# ---------------------------------------------------------------------------
# Loading the raw data from a file or a url
# ---------------------------------------------------------------------------
import urllib.error

import pytest

from src.ingest import pthDownloadFile, strFileFingerprint, tplLoadRawData


class _FakeResponse:
    """A stand-in for what urlopen returns, so no test touches the network."""

    def __init__(self, bytBody):
        self.bytBody = bytBody

    def read(self):
        return self.bytBody

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_an_empty_file_is_reported_by_name(tmp_path):
    """An empty CSV stops with a message that names the file."""
    pthFile = tmp_path / "empty.csv"
    pthFile.write_text("")
    with pytest.raises(ValueError, match="empty.csv"):
        dfLoadCsv(pthFile)


def test_a_header_with_no_rows_is_reported(tmp_path):
    """A CSV with only a header row is treated as no data."""
    pthFile = tmp_path / "header.csv"
    pthFile.write_text("a,b\n")
    with pytest.raises(ValueError, match="no rows"):
        dfLoadCsv(pthFile)


def test_the_fingerprint_is_the_same_for_the_same_contents(tmp_path):
    """Same bytes give the same fingerprint, different bytes do not."""
    (tmp_path / "a.csv").write_text("x\n1\n")
    (tmp_path / "b.csv").write_text("x\n1\n")
    (tmp_path / "c.csv").write_text("x\n2\n")
    assert strFileFingerprint(tmp_path / "a.csv") == strFileFingerprint(tmp_path / "b.csv")
    assert strFileFingerprint(tmp_path / "a.csv") != strFileFingerprint(tmp_path / "c.csv")


def test_a_file_source_reads_raw_path(tmp_path):
    """With source file the CSV at raw_path is read and the load is described."""
    pthRaw = tmp_path / "raw.csv"
    pthRaw.write_text("a,b\n1,2\n3,4\n")
    dfData, dctInfo = tplLoadRawData({"source": "file", "raw_path": str(pthRaw), "raw_url": None,
                                      "download_timeout": 5, "download_retries": 1})
    assert len(dfData) == 2
    assert dctInfo["source"] == "file"
    assert dctInfo["location"] == str(pthRaw)
    assert len(dctInfo["fingerprint"]) == 12


def test_a_url_source_downloads_into_raw_path_then_reads_it(tmp_path, monkeypatch):
    """With source url the address is downloaded to raw_path and that copy is read."""
    monkeypatch.setattr("urllib.request.urlopen", lambda strUrl, timeout: _FakeResponse(b"a,b\n1,2\n"))
    pthRaw = tmp_path / "raw" / "data.csv"
    dfData, dctInfo = tplLoadRawData({"source": "url", "raw_path": str(pthRaw), "raw_url": "https://example.org/d.csv",
                                      "download_timeout": 5, "download_retries": 1})
    assert pthRaw.is_file()
    assert len(dfData) == 1
    assert dctInfo["location"] == "https://example.org/d.csv"


def test_a_failed_download_is_retried_then_reported(tmp_path, monkeypatch, capsys):
    """Every attempt is made, then ingest stops with a message naming the address."""
    intCalls = []

    def subFail(strUrl, timeout):
        intCalls.append(1)
        raise urllib.error.URLError("no route")

    monkeypatch.setattr("urllib.request.urlopen", subFail)
    monkeypatch.setattr("time.sleep", lambda intSeconds: None)
    with pytest.raises(ConnectionError, match="after 3 attempts"):
        pthDownloadFile("https://example.org/d.csv", tmp_path / "d.csv", 5, 3)
    assert len(intCalls) == 3
    assert "attempt 3 of 3 failed" in capsys.readouterr().out


def test_a_retry_that_works_saves_the_file(tmp_path, monkeypatch):
    """One failure followed by a good response still gives the file."""
    lstReplies = [urllib.error.URLError("busy"), _FakeResponse(b"a\n1\n")]

    def subReply(strUrl, timeout):
        varReply = lstReplies.pop(0)
        if isinstance(varReply, Exception):
            raise varReply
        return varReply

    monkeypatch.setattr("urllib.request.urlopen", subReply)
    monkeypatch.setattr("time.sleep", lambda intSeconds: None)
    pthDownloadFile("https://example.org/d.csv", tmp_path / "d.csv", 5, 2)
    assert (tmp_path / "d.csv").read_text() == "a\n1\n"


def test_a_failed_download_leaves_the_existing_copy_alone(tmp_path, monkeypatch):
    """A good local copy is not replaced or damaged by a failed download."""
    pthFile = tmp_path / "d.csv"
    pthFile.write_text("a\n1\n")
    monkeypatch.setattr("urllib.request.urlopen", lambda strUrl, timeout: (_ for _ in ()).throw(urllib.error.URLError("down")))
    monkeypatch.setattr("time.sleep", lambda intSeconds: None)
    with pytest.raises(ConnectionError):
        pthDownloadFile("https://example.org/d.csv", pthFile, 5, 2)
    assert pthFile.read_text() == "a\n1\n"
    assert not (tmp_path / "d.csv.part").exists()


def test_an_empty_download_is_rejected(tmp_path, monkeypatch):
    """A server that answers with nothing is a failure, not a file."""
    monkeypatch.setattr("urllib.request.urlopen", lambda strUrl, timeout: _FakeResponse(b""))
    monkeypatch.setattr("time.sleep", lambda intSeconds: None)
    with pytest.raises(ConnectionError, match="empty"):
        pthDownloadFile("https://example.org/d.csv", tmp_path / "d.csv", 5, 1)


def test_only_http_and_https_addresses_are_downloaded(tmp_path):
    """file: and ftp: addresses are refused."""
    with pytest.raises(ValueError, match="http"):
        pthDownloadFile("file:///etc/passwd", tmp_path / "d.csv", 5, 1)
