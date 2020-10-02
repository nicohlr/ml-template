import pandas as pd


def features_engineering(df):

    # add feature engineering here

    return df


if __name__ == "__main__":

    df = pd.read_csv("../input/train_folds.csv")
    df = features_engineering(df)
    df.to_csv("../input/train_fe_folds.csv", index=False)
