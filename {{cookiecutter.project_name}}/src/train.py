import argparse
import os

import joblib
import pandas as pd
from sklearn import metrics

import config
from models import models
from prepare import prepare


def train(fold, model, last):

    # read the training data with folds
    df = pd.read_csv(config.TRAIN_FILE)

    if last:
        # train on whole dataset
        x_train = df.drop([config.TARGET_LABEL, "FOLD"], axis=1)
        y_train = df[config.TARGET_LABEL]
    else:
        # training data is where kfold is not equal to provided fold
        df_train = df[df["FOLD"] != fold].reset_index(drop=True)

        # validation data is where kfold is equal to provided fold
        df_valid = df[df["FOLD"] == fold].reset_index(drop=True)

        # create training samples
        x_train = df_train.drop([config.TARGET_LABEL, "FOLD"], axis=1)
        y_train = df_train[config.TARGET_LABEL]

        # create validation samples
        x_valid = df_valid.drop([config.TARGET_LABEL, "FOLD"], axis=1)
        y_valid = df_valid[config.TARGET_LABEL]

    # perform cleaning, feature engineering,
    # categorical variables encoding & scaling
    x_train = prepare(x_train)

    # fetch the model from models
    clf = models[model]

    # fit the model on training data
    clf.fit(x_train, y_train)

    if not last:
        # calculate & print metric
        predictions = clf.predict(prepare(x_valid))
        scorer = metrics.get_scorer(config.METRIC)
        metric = scorer._score_func(y_valid, predictions)
        print(f"Fold={fold}, {config.METRIC}={metric}")

    model_path = f"{model}.bin" if last else f"{model}_fold{fold}.bin"

    # save the model
    joblib.dump(clf, os.path.join(config.MODEL_OUTPUT, model_path))

    if last:
        print("Last model saved at: " + model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(*["--fold", "-f"], type=str, default=None)
    parser.add_argument(*["--model", "-m"], type=str)

    args = parser.parse_args()

    if args.fold == "all":
        for f in range(config.N_FOLDS):
            train(fold=f, model=args.model, last=args.last)
    elif args.fold is None:
        train(fold=args.fold, model=args.model, last=True)
    else:
        train(fold=int(args.fold), model=args.model, last=False)
