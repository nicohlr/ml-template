import pandas as pd


def _cleaning(df):

    # add cleaning here
    # for example, replace NaN values

    return df


def _feature_engineering(df):

    # add feature engineering here
    # for example, create new columns

    return df


def _categorical_encoding(df):

    # add categorical variable encoding here
    # for example, use a OneHotEncoder

    return df


def _scaling(df):

    # add scaling here
    # for example, use a StandarScaler

    return df


def prepare(df):

    df = _cleaning(df)
    df = _feature_engineering(df)
    df = _categorical_encoding(df)
    df = _scaling(df)

    return df
