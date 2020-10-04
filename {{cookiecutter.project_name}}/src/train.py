import argparse
import os

import joblib
import pandas as pd
from sklearn import metrics

import config
import models
from prepare import prepare


def train(fold, model, final):

    # read the training data with folds
    df = pd.read_csv(config.TRAIN_FILE)

    if final:
        # train on whole dataset
        x_train = df.drop(config.TARGET_LABEL, axis=1)
        y_train = df[config.TARGET_LABEL]
    else:
        # training data is where kfold is not equal to provided fold
        df_train = df[df['fold'] != fold].reset_index(drop=True)

        # validation data is where kfold is equal to provided fold
        df_valid = df[df['fold'] == fold].reset_index(drop=True)

        # create training samples
        x_train = df_train.drop(config.TARGET_LABEL, axis=1)
        y_train = df_train[config.TARGET_LABEL]

        # create validation samples
        x_valid = df_valid.drop(config.TARGET_LABEL, axis=1)
        y_valid = df_valid[config.TARGET_LABEL]

    # perform cleaning, feature engineering,
    # categorical variables encoding & scaling
    x_train = prepare(x_train)

    # fetch the model from models
    clf = models.models[model]

    # fit the model on training data
    clf.fit(x_train, y_train)

    if not final:
        # calculate & print metric
        predictions = clf.predict(prepare(x_valid))
        accuracy = metrics.accuracy_score(y_valid, predictions)
        print(f'Fold={fold}, Accuracy={accuracy}')

    # save the model
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, f'{model}_fold{fold}.bin')
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--fold',
        type=str,
        default='all'
    )
    parser.add_argument(
        '--model',
        type=str
    )
    parser.add_argument(
        '--final',
        type=bool,
        default=False
    )

    args = parser.parse_args()

    if args.fold == 'all':
        for f in range(config.N_FOLDS):
            train(fold=f, model=args.model, final=args.final)
    else:
        train(
            fold=int(args.fold),
            model=args.model,
            final=args.final
        )
