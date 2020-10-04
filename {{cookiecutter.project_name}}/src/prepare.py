import pandas as pd


def cleaning(df):

    # add cleaning here
    # for example, replace NaN values

    return df


def feature_engineering(df):

    # add feature engineering here
    # for example, create new columns

    return df


def categorical_encoding(df):

    # add categorical variable encoding here
    # for example, use a OneHotEncoder

    return df


def scaling(df):

    # add scaling here
    # for example, use a StandarScaler

    return df


def prepare(df):

    df = cleaning(df)
    df = feature_engineering(df)
    df = categorical_encoding(df)
    df = scaling(df)

    return df
