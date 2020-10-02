import argparse
import fasttext
import pandas as pd

import config


def run(model_path):

    # read the test data
    df = pd.read_csv(config.TEST_FILE)

    # add standard scaling / categorical features encoding here
    # the transformations must be identical to those made on the training set

    # fetch the pretrained model
    clf = fasttext.load_model(model_path)

    # predict on test dataset
    predictions = clf.predict(df)
    df['predictions'] = predictions

    # dump final dataset with predictions
    df.to_csv('../predictions.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_path',
        type=str,
    )

    args = parser.parse_args()

    run(
        model_path=args.model_path
    )
