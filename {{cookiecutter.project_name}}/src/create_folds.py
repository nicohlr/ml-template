import config

import pandas as pd
from sklearn import model_selection


if __name__ == "__main__":

    df = pd.read_csv("../input/train.csv")
    df = df.dropna().reset_index(drop=True)
    df["FOLD"] = -1

    # shuffle dataset
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch target
    y = df[config.TARGET_LABEL]

    kf = model_selection.StratifiedKFold(n_splits=config.N_FOLDS)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "FOLD"] = f

    df.to_csv("../input/train_folds.csv", index=False)
    print("Training set with folds saved at: input/train_folds.csv")
