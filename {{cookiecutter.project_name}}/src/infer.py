import argparse
import joblib
import pandas as pd

import config
from prepare import prepare


def infer(model_path):

    # read the test data
    df = pd.read_csv(config.TEST_FILE)

    # perform cleaning, feature engineering,
    # categorical variables encoding & scaling
    df = prepare(df)

    # fetch the pretrained model
    clf = joblib.load(model_path)

    # predict on test dataset
    predictions = clf.predict(df)
    df['predictions'] = predictions

    # dump dataset with predictions
    df.to_csv('../submission.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        *['--model_path', '-mp'],
        type=str,
    )

    args = parser.parse_args()

    infer(
        model_path=args.model_path
    )
