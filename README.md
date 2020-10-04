# ml-template

A cookiecutter template for any ML/DL project.

## Usage

If you don't have cookiecutter, install it:

    pip install cookiecutter

Then, run the following command line. A complete project folder will automatically be created in current path:

    cookiecutter https://github.com/nicohlr/ml-template

This project folder contains subfolders, in which you can store your data, notebooks, templates and scripts. Start by replacing the default datasets in the input folder by your data. Then run the **create_folds.py** script:

    python create_folds.py

This will create a *train_folds.csv* file in input folder. This file is the train dataset shuffled with an additionnal "fold" column for k-fold CV. Finally, you can train you model easily on the fold of your choice (model is automatically saved in models folder) with the **train.py** script:

    python train.py --fold 1 --model lr

**Note:** setting the --fold argument to `all` will train sequentially on all folds.

You can use other machine learning models for training by referencing them in the **models.py** script. You can also add cleaning, feature engineering, categorical variables encoding and scaling steps that will be applied on the data before training by filling the functions of the **prepare.py** script.

When you are ready to make the final training, just run the following command:

    python train.py --final True --model YOUR_MODEL

**Note:** When final argument is set to `True`, the fold argument is ignored and the training is performed on the whole training set.

Finally, you can run inference:

    python inference.py --model_path PATH_TO_FINAL_MODEL
