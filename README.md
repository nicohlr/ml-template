# ml-template

A cookiecutter template for any ML/DL project.

## Create project

If you don't have cookiecutter, install it:

    pip install cookiecutter

Then, run the following command line:

    cookiecutter https://github.com/nicohlr/ml-template

You will be asked for some information about your project:

- **PROJECT_NAME**: The name of the project folder that will be created.
- **TARGET_LABEL**: The label of your target column.
- **METRIC**: The metric that will be used to evaluate models. You must choose a metric among [the list of available metrics in scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values). You must choose a metric that suits your problem (don't choose a regression metric if your problem is a classification ...).

A complete project folder will then be created in current path. This project folder contains subfolders, in which you can store your data, notebooks, templates and scripts:
 
```
├── LICENSE
├── README.md              <- The top-level README
├── input                  <- You can store your data here
│   ├── train.csv
│   └── test.csv
├── models                 <- All trained models
├── notebooks              <- You can store your notebooks here
├── src                    <- Source code for use in this project.
│   ├── config.py          <- Useful variables used in scripts
│   ├── create_folds.py    <- Create a training dataset for CV
│   ├── infer.py           <- Run inference on test dataset
│   ├── models.py          <- Reference all models here
│   ├── prepare.py         <- Code for cleaning, FE, scaling ...
│   └── train.py           <- Run training
└── requirements.txt       <- All requirements of the project
```

## Training

Start by replacing the default datasets in the input folder by your data. Then run the **create_folds.py** script:

    python create_folds.py

This will create a *train_folds.csv* file in input folder. This file is the train dataset shuffled with an additionnal "FOLD" column that will be used for k-fold cross validation. Then, you can evaluate a model on the fold of your choice with the **train.py** script. Training is then performed on all the remaining folds.

For example, to train a model on folds 1 to 4 and evaluate its performance on the fold 0 you must run:

    python train.py --fold 0 --model lr

**Note:** setting the --fold argument to `all` will evaluate sequentially on each fold.

You can use other machine learning models for training by referencing them in the **models.py** script. You can also add cleaning, feature engineering, categorical variables encoding and scaling steps that will be applied on the data before training by filling the functions of the **prepare.py** script.

**Note:** At each training, the resulting model is automatically saved in models folder.

## Inference

When you are ready to make the last training (train on the whole training set), just run the following command:

    python train.py --model YOUR_MODEL

**Note:** When the fold argument is not specified, the training is performed on the whole training set.

Finally, you can run inference:

    python infer.py --model_path ../models/YOUR_FINAL_MODEL.bin

When inference is finished, a *submission.csv* file, containing predictions on test set, is dumped at the root of the projet.