"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS
    check_setup.py

    Reports what is installed for the pipeline and prints one pip command for
    whatever is missing. Run it from inside the activated virtual environment:
        python check_setup.py
"""

import importlib.metadata as objMeta
import shutil
import sys

# Packages from requirements.txt. The code and tests need these.
LST_CORE = ["pandas", "scikit-learn", "joblib", "pyyaml", "pytest", "dvc"]

# Packages the slide deck's extended pipeline adds.
LST_DECK = ["mlflow-skinny", "sqlalchemy", "alembic", "scipy", "fastapi", "uvicorn",
            "pydantic", "httpx"]

# Programs that are installed on the computer, not with pip.
LST_PROGRAMS = ["git", "docker"]


def strVersionOf(strPackage: str) -> str:
    """Return the installed version of a package, or an empty string if missing."""
    try:
        return objMeta.version(strPackage)
    except objMeta.PackageNotFoundError:
        return ""


def lstReport(strTitle: str, lstPackages: list[str]) -> list[str]:
    """Print one line per package and return the names that are missing."""
    print(f"\n{strTitle}")
    lstMissing = []
    for strPackage in lstPackages:
        strVersion = strVersionOf(strPackage)
        if strVersion:
            print(f"  installed  {strPackage:14} {strVersion}")
        else:
            print(f"  MISSING    {strPackage}")
            lstMissing.append(strPackage)
    return lstMissing


def subMain() -> None:
    print(f"Python {sys.version.split()[0]} at {sys.executable}")
    lstMissing = lstReport("In requirements.txt (needed now):", LST_CORE)
    lstMissing += lstReport("From the deck (needed for the full pipeline):", LST_DECK)

    print("\nPrograms (not installed with pip):")
    for strProgram in LST_PROGRAMS:
        strPath = shutil.which(strProgram)
        print(f"  {'installed ' if strPath else 'MISSING   '} {strProgram:14} {strPath or ''}")

    if lstMissing:
        print("\nTo install what is missing:\n  pip install " + " ".join(lstMissing))
    else:
        print("\nAll packages are installed.")


if __name__ == "__main__":
    subMain()
