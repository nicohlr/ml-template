# ml-template

A cookiecutter template for any ML/DL project.

## Usage

If cookiecutter is not installed, run:

    pip install cookiecutter

Then, run the following command line and a complete project folder will automatically be created in current path:

    cookiecutter https://github.com/nicohlr/ml-template

This project folder contains subfolders, in which you can store your data, notebooks, templates and scripts. Start by replacing the placeholders datasets in the input folder by your data. Then run:

    python create_folds.py

This will create a train_folds.csv file in input folder. This file is the train dataset shuffled with an additionnal "fold" column for k-fold CV. Then, you can train you model easily on the fold of your choice (model is automatically saved in models folder):

    python train.py --fold 1 --model rf

You can use other models for passing to the model argument by adding it to the models.py script. 