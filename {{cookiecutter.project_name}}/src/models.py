from sklearn import linear_model
from sklearn import ensemble

models = {
    'lr': linear_model.LogisticRegression(),
    'rf': ensemble.RandomForestClassifier()
}
