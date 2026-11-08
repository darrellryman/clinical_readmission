"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    display.py

    Shared output helpers for the pipeline.

    This module holds the text styles, the function that prints a profile of a
    DataFrame, and the helpers that send a stage's printed output to an RTF file
    instead of the screen. It lives in its own file because more than one stage
    needs it (for example, ingest.py prints a profile after loading and again
    after cleaning, and every stage saves its output to RTF).

    The RTF file name comes from config.yaml (outputs.rtf_path, or
    testing.rtf_path when tests run). The text {stage} in that name is replaced
    with the stage name, so each stage gets its own file.

    Import only what a stage uses:
        from src.display import objCaptureToRtf, bDisplayDatasetDetails
"""

# contextlib lets us send print() output somewhere else for a block of code.
import contextlib

# datetime stamps each RTF file with the time of the run.
import datetime

# io.StringIO is an in-memory "file" that catches everything printed.
import io

# re finds the bold, italic and underline codes so they can become RTF styles.
import re

# struct reads the width and height out of a PNG file, and sys.stdout tells
# us whether the output is being caught for an RTF file.
import struct
import sys

# Path handles file paths the same way on Windows, macOS and Linux.
from pathlib import Path

# pandas builds the small summary tables (missing and negative counts).
import pandas as pd

# ANSI codes that make terminal text bold, italic or underlined. RESET switches
# the styles off again. Modern Windows terminals, Git Bash, macOS and Linux show them.
STR_BOLD = "\033[1m"
STR_ITALIC = "\033[3m"
STR_UNDERLINE = "\033[4m"
STR_RESET = "\033[0m"
STR_LARGE = "\033[26m"

# Markers that wrap a picture's file name and width in the printed text. subPrintImage()
# writes them, and pthSaveRtf() swaps each pair for the picture itself. They are
# control characters that never appear in normal text.
STR_IMAGE_START = "\x02"
STR_IMAGE_END = "\x03"

# What each ANSI style number becomes in an RTF file. 0 switches everything off.
DCT_RTF_STYLES = {
    "1": "\\b ",
    "3": "\\i ",
    "4": "\\ul ",
    "0": "\\b0\\i0\\ulnone ",
    "": "\\b0\\i0\\ulnone ",
}

def subPrintPiplineHeader(strPipelineStage: str = "", strClientQuestion: str = "", strResponseVariable: str = "", strPredictors: str = "", strDataset: str = "") -> None:
    """Print the pipline header in .

    Call it in a stage, just below the line of code that gives the evidence for
    the step. Because it uses print(), the text goes to the stage's RTF file
    with everything else. Write the answer between the quotes of strAnswer. Use
    triple quotes if the answer runs over several lines.

    Args:
        strPipelineStage: The name of the pipeline stage. If it is blank, it does not print the header.

    Returns:
        None. It only prints.
    """
    # A blank parameter would look like a finished step with nothing in it.
    strPipelineStage = strPipelineStage.strip() or ""
    strClientQuestion = strClientQuestion.strip() or ""
    strResponseVariable = strResponseVariable.strip() or ""
    strPredictors = strPredictors.strip() or ""
    strDataset = strDataset.strip() or ""

    ##  Assessment 2: Applied MLOps pipeline project
    print(f"\n{STR_BOLD}###################################################################################{STR_RESET}")
    print(f"\n")
    print(f"{STR_LARGE}{STR_BOLD}Assessment 2: Applied MLOps pipeline project{STR_RESET}")
    print(f"\n")
    print(f"\n{STR_BOLD}Name:{STR_RESET} Darrell Ryman")
    print(f"\n{STR_BOLD}Student ID:{STR_RESET} S00445230")
    print(f"\n{STR_BOLD}Course:{STR_RESET} AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS")
    print(f"\n")
    print(f"\n{STR_BOLD}Pipeline Stage:{STR_RESET} {strPipelineStage}")
    print(f"\n")
    print(f"\n{STR_BOLD}Client question:{STR_RESET} {strClientQuestion}")
    print(f"\n{STR_BOLD}Response variable:{STR_RESET} {strResponseVariable}")
    print(f"\n{STR_BOLD}Predictors:{STR_RESET} {strPredictors}")
    print(f"\n{STR_BOLD}Data source:{STR_RESET} {strDataset}")
    print(f"\n")
    print(f"\n{STR_BOLD}##################################################################################{STR_RESET}\n")


def subPrintStep(strQuestion: str, strAnswer: str = "") -> None:
    """Print one assessment step and its answer, in bold italic, so it stands out.

    Call it in a stage, just below the line of code that gives the evidence for
    the step. Because it uses print(), the text goes to the stage's RTF file
    with everything else. Write the answer between the quotes of strAnswer. Use
    triple quotes if the answer runs over several lines.

    Args:
        strQuestion: The step, starting with its number, for example
            "Step 1 - Define the problem...".
        strAnswer: Your answer. If it is blank, the report says the answer has
            not been written yet, so a missing answer is easy to spot.

    Returns:
        None. It only prints.
    """
    # A blank answer would look like a finished step with nothing in it, so say so.
    strShown = strAnswer.strip() or "(answer not written yet)"

    # Bold plus italic on, the text, then everything off again.
    print(f"\n{STR_ITALIC}{strQuestion}{STR_RESET}")
    print(f"\n{STR_BOLD}{STR_ITALIC}Answer:{STR_RESET} {STR_ITALIC}{strShown}{STR_RESET}\n")


def bDisplayDatasetDetails(dfData) -> bool:
    """Print a profile of a DataFrame so you can see its condition at a glance.

    Shows the row, column and cell counts, the data types, the column headers,
    missing values (overall, by row, by column and per column), negative values
    per numeric column, duplicate rows, summary statistics, the dataset info
    and the first five rows. Call it after loading and again after cleaning,
    then compare the two printouts.

    Args:
        dfData: The DataFrame to describe. Callers get it from dfLoadCsv().
            None is allowed and is reported as an error.

    Returns:
        True if the details were printed. False if dfData is None, in which
        case an error message is printed and nothing else happens.
    """
    if dfData is not None:
        # ##########################        DATASET SUMMARY     ###################################### 
        # Display the number of rows (observations) and columns (attributes)
        print(f"\n{STR_BOLD}{STR_UNDERLINE}Data description{STR_RESET}")
        print(f"{STR_BOLD}Rows:{STR_RESET} {dfData.shape[0]}")
        print(f"{STR_BOLD}Columns:{STR_RESET} {dfData.shape[1]}")

        # Display the total number of cells in the dataset
        print(f"{STR_BOLD}Cells:{STR_RESET} {dfData.size}")

        # Display total number of data type
        print(f"\n{STR_BOLD}{STR_UNDERLINE}Total number for each data type:{STR_RESET}")
        print(dfData.dtypes.value_counts())

        # Display the formatted column headers 
        print(f"\n{STR_BOLD}{STR_UNDERLINE}The column headers:{STR_RESET}")
        print(dfData.columns)

        # ##########################     MISSING VALUES SUMMARY     ################################## 
        # Missing value summary 
        print(f"\n{STR_BOLD}{STR_UNDERLINE}Missing values summary{STR_RESET}")
        print(f"\n{STR_BOLD}Total missing values in the whole dataset :{STR_RESET} {dfData.isna().sum().sum()}")

        # Percentage of the total number of missing values in the dataset
        print(
            f"{STR_BOLD}Percentage total missing values:{STR_RESET}{round((dfData.isna().sum().sum() / dfData.size) * 100, 2)}%")

        # Percentage of rows with at least one missing value
        fltRowsMissing = (dfData.isna().any(axis=1).sum() / len(dfData)) * 100
        print(f"{STR_BOLD}% of rows with at least one missing value:{STR_RESET} {round(fltRowsMissing, 2)}%")

        # Percentage of columns with at least one missing value
        fltColumnsMissing = (dfData.isna().any(axis=0).sum() / dfData.shape[1]) * 100
        print(f"{STR_BOLD}% of columns with at least one missing value:{STR_RESET} {round(fltColumnsMissing, 2)}%")

        # Number and percentage of missing values per column
        print(f"\n{STR_BOLD}{STR_UNDERLINE}Number and % of missing values per column{STR_RESET}")

        # dfMissing = pandas DataFrame object
        dfMissing = pd.DataFrame({'Missing Count': dfData.isna().sum(),
                                            'Missing %': (dfData.isna().mean() * 100).round(2)})
        print(dfMissing)

        # ##########################        NEGATIVE VALUES SUMMARY        ########################## 
        # Number and percentage of negative values per numeric column
        print(f"\n{STR_BOLD}{STR_UNDERLINE}Number and % of negative values per numeric column{STR_RESET}")

        # lstNumericColumns = list of the numeric column names
        lstNumericColumns = dfData.select_dtypes(include='number').columns.tolist()

        # dfNegative = pandas DataFrame object
        dfNegative = pd.DataFrame({'Negative Count': (dfData[lstNumericColumns] < 0).sum(),
                                             'Negative %': ((dfData[lstNumericColumns] < 0).mean() * 100).round(2)})

        # Print the summary of the negative values
        print(dfNegative)

        # ##########################        DUPLICATE ROWS SUMMARY      ############################## 
        # Print the duplicated Rows 
        print(f"\n{STR_BOLD}{STR_UNDERLINE}Duplicate Rows{STR_RESET}")
        print(f"{STR_BOLD}Number of duplicated Rows:{STR_RESET} {dfData.duplicated().sum()}")
        print(dfData[dfData.duplicated(keep=False)].head(10))

        # ##########################        DATASET STATISTICS SUMMARY      ########################## 
        # Display the Count, mean, std, min, max, and quartiles 
        print(f"\n{STR_BOLD}{STR_UNDERLINE}Count, mean, std, min, max, quartiles for numeric columns:{STR_RESET}")
        print(dfData.describe().round(2))

        # ##########################        DATASET DETAIL INFORMATION     ########################### 
        # Display the dataset information
        print(f"\n{STR_BOLD}{STR_UNDERLINE}Dataset information{STR_RESET}")
        dfData.info()

        # Return the first 5 rows as a sanity check
        print(f"\n{STR_BOLD}{STR_UNDERLINE}The first 5 rows as a sanity check:{STR_RESET}")
        print(dfData.head())

        return True

    else:
        print(f"{STR_BOLD}Error:{STR_RESET} Dataset is NULL.")
        return False


def strBuildRtfImage(varPath, fltWidthInches: float = 6.0) -> str:
    """Turn a PNG file into the RTF text that shows it as a picture.

    Word and WordPad read the picture straight from the RTF file, so the file is
    self-contained and needs no image files beside it. The picture is scaled to
    the width asked for, keeping its shape.

    Args:
        varPath: The PNG file, as text or a Path.
        fltWidthInches: How wide the picture is on the page. The text area of a
            default RTF page is about 6 inches wide.

    Returns:
        One line of RTF with no line breaks in it. The picture data is written as hex.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a PNG.
    """
    bytPng = Path(varPath).read_bytes()
    if bytPng[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {varPath}")

    # A PNG stores its size in pixels at fixed positions near the start of the file.
    intWidth, intHeight = struct.unpack(">II", bytPng[16:24])
    intGoalWidth = int(fltWidthInches * 1440)
    intGoalHeight = int(intGoalWidth * intHeight / intWidth)

    # \picw and \pich are the pixel size. \picwgoal and \pichgoal are the size on the
    # page in twips (1440 to the inch). No line breaks, because pthSaveRtf turns each one into \par.
    return (f"{{\\pict\\pngblip\\picw{intWidth}\\pich{intHeight}"
            f"\\picwgoal{intGoalWidth}\\pichgoal{intGoalHeight} {bytPng.hex()}}}")


def subPrintImage(varPath, fltWidthInches: float = 6.0) -> None:
    """Print a picture, so it appears in the stage's RTF file with the rest of the output.

    Call it like print(), inside a stage. When the output is being saved to an RTF
    file, the picture is embedded in the file. When it is going to the screen, the
    file name is printed instead, because a terminal can't show a picture.

    Args:
        varPath: The PNG file, as text or a Path.
        fltWidthInches: How wide the picture is on the page, in inches.

    Returns:
        None. It only prints.
    """
    # objCaptureToRtf() sends the output to an io.StringIO, so that is how we know it is being caught.
    if isinstance(sys.stdout, io.StringIO):
        print(f"\n{STR_IMAGE_START}{varPath}|{fltWidthInches}{STR_IMAGE_END}\n")
    else:
        print(f"\nImage: {varPath}\n")


def pthResolveRtfPath(varTemplate, strStage: str) -> Path:
    """Turn the configured RTF name into a real path for one stage.

    Args:
        varTemplate: The path from config.yaml, as text or a Path. The text
            {stage} in it is replaced by the stage name, so
            "reports/{stage}_output.rtf" becomes "reports/ingest_output.rtf".
        strStage: The stage name, for example "ingest".

    Returns:
        The path for that stage's RTF file.
    """
    # Lower case, so "Ingest" and "ingest" give the same file name on every system.
    return Path(str(varTemplate).replace("{stage}", strStage.lower()))


def pthSaveRtf(strText: str, varPath) -> Path:
    """Write text to an RTF file, creating the folder if needed.

    Args:
        strText: The text to save. Line breaks are kept.
        varPath: Where to write the file, as text or a Path.

    Returns:
        The path that was written.
    """
    # Split the text at the terminal style codes (bold, italic, underline). The
    # codes are kept in the list so each one can become its RTF twin.
    lstPieces = re.split(r"(\x1b\[[0-9;]*m)", strText)

    lstOut = []
    for strPiece in lstPieces:
        if strPiece.startswith("\x1b["):
            # A style code such as ESC[1;3m. Turn each number into an RTF style.
            for strNumber in strPiece[2:-1].split(";"):
                lstOut.append(DCT_RTF_STYLES.get(strNumber, ""))
        else:
            # A text piece may hold picture markers from subPrintImage(). Each one becomes
            # the picture. Everything between them is ordinary text.
            for strSub in re.split(f"({STR_IMAGE_START}[^{STR_IMAGE_END}]*{STR_IMAGE_END})", strPiece):
                if strSub.startswith(STR_IMAGE_START):
                    strFile, strWidth = strSub[1:-1].rsplit("|", 1)
                    try:
                        lstOut.append(strBuildRtfImage(strFile, float(strWidth)))
                    except (OSError, ValueError) as objError:
                        # A missing or bad picture must not stop the report. Say so in its place.
                        lstOut.append(f"(picture not shown: {Path(strFile).name}, {type(objError).__name__})")
                    continue

                # RTF treats \ { } as commands, so each one needs a backslash in front.
                strEscaped = strSub.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")

                # Characters outside plain ASCII are written as \uNNNN? codes.
                lstOut.append("".join(strChar if ord(strChar) < 128 else f"\\u{ord(strChar)}?" for strChar in strEscaped))
    strEscaped = "".join(lstOut)

    # \par is the RTF line break. Courier New keeps table columns lined up.
    strBody = strEscaped.replace("\n", "\\par\n")
    strRtf = "{\\rtf1\\ansi\\deff0{\\fonttbl{\\f0 Courier New;}}\\f0\\fs18\n" + strBody + "}"

    pthFile = Path(varPath)
    pthFile.parent.mkdir(parents=True, exist_ok=True)
    pthFile.write_text(strRtf, encoding="ascii")
    return pthFile


@contextlib.contextmanager
def objCaptureToRtf(varRtfPath, strStage: str):
    """Send everything printed inside the block to an RTF file, not the screen.

    Use it as a with block. When the block ends, the captured text is saved and
    one line saying where it went is printed to the screen. The file is saved
    even if the block stops with an error, so you can see what ran before it.
    If the file cannot be written (for example it is open in Word, so it is
    locked), a warning is printed on the screen and the stage carries on.

    Args:
        varRtfPath: The RTF path from config.yaml, with {stage} in it. If it is
            blank or None, nothing is captured and the output goes to the
            screen as normal.
        strStage: The stage name, used to fill in {stage} and to label the file.

    Yields:
        The path of the RTF file, or None when output is going to the screen.
    """
    # A blank path means "print to the screen", so there is nothing to capture.
    if not varRtfPath:
        yield None
        return

    pthFile = pthResolveRtfPath(varRtfPath, strStage)
    objBuffer = io.StringIO()
    try:
        # While inside this block, print() writes into objBuffer, not the screen.
        with contextlib.redirect_stdout(objBuffer):
            yield pthFile
    finally:
        strStamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            pthSaveRtf(f"{STR_BOLD}{strStage} output:{STR_RESET} {strStamp}\n\n" + objBuffer.getvalue(), pthFile)
            print(f"Output saved to {pthFile}")
        except PermissionError:
            # Windows refuses the write while Word or WordPad has the file open.
            print(f"{STR_BOLD}WARNING:{STR_RESET} {pthFile} was NOT updated. "
                  f"The file is locked or access was denied. "
                  f"Close it in Word or WordPad and run the stage again.")
        except OSError as objError:
            # Any other problem writing the file, such as a bad path or a full disk.
            print(f"{STR_BOLD}WARNING:{STR_RESET} {pthFile} was NOT updated. "
                  f"The file could not be written ({objError}).")
