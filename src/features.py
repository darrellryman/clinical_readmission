"""
    Assessment 2: Applied MLOps pipeline project

    Name: Darrell Ryman

    Student ID: S00445230

    Course: AINT602 - AI SYSTEMS ENGINEERING AND MACHINE LEARNING OPERATIONS

    features.py
    
    Stage 3 of the pipeline: feature building and data splitting.
    
    Turns a validated DataFrame into the two things a model needs:
        dfX:  the input columns (features)
        serY: the answer column (target)
    
    It also splits the data into a training set and a test set. The split lives in
    this one place so that train.py and evaluate.py always agree on which rows are
    training rows and which are test rows.
    
    It also holds dfEngineerFeatures(), which builds the calculated and encoded
    columns, and objBuildFeatureStep(), which wraps that function as the first
    step of the saved pipeline. The saved model file therefore carries its own
    feature step, so training, evaluation, prediction and the web service all
    prepare the columns the same way (no train-serve mismatch).

    The column names, test size and random seed all come from config.yaml.
"""

import pandas as pd

# FunctionTransformer wraps a plain function as a pipeline step.
from sklearn.preprocessing import FunctionTransformer

# train_test_split shuffles the rows and divides them into two groups.
from sklearn.model_selection import train_test_split


def dfEngineerFeatures(dfX: pd.DataFrame, dctConfig: dict) -> pd.DataFrame:
    """Add the calculated columns and encode the text columns, the same way every time.

    Two kinds of change are made, both set in config.yaml:
        1. Difference columns (features.difference_columns): a new column that is
           one column minus another, for example pulse_pressure = systolic_bp
           minus diastolic_bp.
        2. Category columns (features.categorical_columns): a text column such as
           sex is replaced by 0/1 columns, one for each category except the first.
           The categories come from the config, not from the data, so training and
           prediction always produce the same columns even if a batch of new
           patients happens to contain only one category.

    Nothing here is learned from the data (no averages or scaling), so it is safe
    to apply to the training rows, the test rows and new patients alike without
    leaking information between them.

    Args:
        dfX: The input columns, as picked by tplBuildFeatures() or dfPredict().
        dctConfig: The loaded config.yaml settings.

    Returns:
        A new DataFrame with the engineered columns. dfX is not changed.

    Raises:
        ValueError: If a category column holds a value that is not on its list
            in the config (a blank counts). Stopping is safer than quietly
            scoring the row as if it were the first category.
    """
    dfOut = dfX.copy()
    dctFeatures = dctConfig["features"]

    # 1. Difference columns: new = first column minus second column.
    for strNew, lstPair in (dctFeatures["difference_columns"] or {}).items():
        dfOut[strNew] = dfOut[lstPair[0]] - dfOut[lstPair[1]]

    # 2. Category columns: one 0/1 column per category, leaving out the first one.
    #    The first category is the reference. If every new column is 0, the row is that one.
    for strColumn, lstCategories in (dctFeatures["categorical_columns"] or {}).items():
        # Find any value that is not on the allowed list. isin() is False for blanks too.
        serUnknown = dfOut.loc[~dfOut[strColumn].isin(lstCategories), strColumn]
        if len(serUnknown) > 0:
            lstUnknown = sorted(set(map(str, serUnknown.unique())))
            raise ValueError(
                f"Column '{strColumn}' has values not in the allowed list {lstCategories}: {lstUnknown}"
            )
        for strCategory in lstCategories[1:]:
            dfOut[f"{strColumn}_{strCategory}"] = (dfOut[strColumn] == strCategory).astype(int)
        dfOut = dfOut.drop(columns=[strColumn])

    return dfOut


def objBuildFeatureStep(dctConfig: dict) -> FunctionTransformer:
    """Wrap dfEngineerFeatures as a pipeline step, so it is saved inside the model file.

    The step learns nothing from the data. It only applies the fixed rules in the
    config, so fitting it on training rows and applying it to new rows give the
    same result as calling dfEngineerFeatures directly.

    Args:
        dctConfig: The loaded config.yaml settings.

    Returns:
        A scikit-learn step that turns the raw feature columns into model columns.
    """
    # validate=False keeps the DataFrame as it is, so column names reach the model.
    # feature_names_out="one-to-one" is not used because the columns change.
    return FunctionTransformer(dfEngineerFeatures, kw_args={"dctConfig": dctConfig}, validate=False)


def dfRawInputs(dfData: pd.DataFrame, dctConfig: dict) -> pd.DataFrame:
    """Return only the raw feature columns, in the config's order, with no changes.

    These are the columns the saved pipeline expects. The pipeline builds the
    calculated and encoded columns itself.

    Args:
        dfData: A validated DataFrame, or new patients.
        dctConfig: The loaded config.yaml settings.

    Returns:
        A new DataFrame holding features.feature_columns.
    """
    return dfData[dctConfig["features"]["feature_columns"]].copy()


def tplBuildFeatures(dfData: pd.DataFrame, dctConfig: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into model inputs (dfX) and the target (serY).

    The inputs are run through dfEngineerFeatures(), which adds the calculated
    columns and encodes the text columns. Prediction calls the same function.

    Args:
        dfData: A validated DataFrame.
        dctConfig: The loaded config.yaml settings.

    Returns:
        A tuple of (dfX, serY).
    """
    # Which columns are inputs and which one is the target is set in the config.
    lstFeatureColumns = dctConfig["features"]["feature_columns"]
    strTargetColumn = dctConfig["features"]["target_column"]

    # .copy() so later changes to dfX or serY can't quietly change the original dfData.
    dfX = dfEngineerFeatures(dfData[lstFeatureColumns], dctConfig)
    serY = dfData[strTargetColumn].copy()
    return dfX, serY


def lstSplitData(dfX: pd.DataFrame, serY: pd.Series, dctConfig: dict):
    """Split features and target into training and test sets.

    Args:
        dfX: The input columns.
        serY: The target column.
        dctConfig: The loaded config.yaml settings. Uses features.test_size (the
            fraction of rows held back for testing), features.stratify (whether
            to keep the class mix the same in both sets) and project.random_state.

    Returns:
        A list of four items: dfXTrain, dfXTest, serYTrain, serYTest (in that order).
    """
    # A fixed random_state makes the shuffle repeatable. Without it, train.py
    # and evaluate.py would each get a different random split, and evaluate.py
    # could end up scoring the model on rows it was trained on.
    #
    # stratify=serY makes both sets keep the same share of each target value. With
    # about 14% readmitted, a plain random split can leave the train and test sets
    # with different rates. None switches this off.
    varStratify = serY if dctConfig["features"]["stratify"] else None
    return train_test_split(
        dfX,
        serY,
        test_size=dctConfig["features"]["test_size"],
        random_state=dctConfig["project"]["random_state"],
        stratify=varStratify,
    )
