# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import model_selection
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_validate
from nltk.stem.porter import *
from sklearn.metrics import f1_score
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import plot_roc_curve
#import warnings

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression

def read_data(filename):
    """
    Takes in a filename returns features and targets numpy array
    """
    df = pd.read_csv(filename)
    X = np.array(df["question_text"])
    y = np.array(df["target"])
    return X, y


from string import punctuation


class StemmerProcess(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
        self.stemmer = PorterStemmer()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X  = [word_tokenize(string) for string in X]
        X = self.remove_punctuation_list(X)
        X = [[self.stemmer.stem(word) for word in string] for string in X]
        X = [" ".join(token for token in list) for list in X]
        return X
    
    def remove_punctuation_list(self, X):
        return [[remove_punctuation(token) for token in string if len(token)>1] for string in X]

def remove_punctuation(word):
    return ''.join(c for c in word if c not in punctuation)
    

class BasicPreProcessing(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X  = [word_tokenize(string) for string in X]
        X = self.remove_punctuation_list(X)
        X = [" ".join(token for token in list) for list in X]
        return X

    def remove_punctuation_list(self, X):
        return [[remove_punctuation(token) for token in string if len(token)>1] for string in X]
        

#add other methods as desired
#Class inspired by https://stackoverflow.com/questions/50285973/pipeline-multiple-classifiers
class modelSwitcher(BaseEstimator):
    def __init__(self, estimator=LogisticRegression()):
        self.estimator = estimator

    def fit(self, X_train, y_train=None, **kwargs):
        self.estimator.fit(X_train,y_train)
        return self

    def predict(self, X_test, y=None):
        return self.estimator.predict(X_test)

    def predict_proba(self, X_test):
        return self.estimator.predict_proba(X_test)

    def score(self, X, y):
        return self.estimator.score(X, y)
X,y = read_data("../input/quora-insincere-questions-classification/train.csv")
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state=0)
print(len(X_train), len(X_test))


Best_LR = make_pipeline(StemmerProcess(), CountVectorizer(ngram_range=(1,3)), LogisticRegression(C=1, solver="liblinear", class_weight="balanced",penalty="l2", max_iter=200, verbose=1, n_jobs=-1))
Best_MNB = make_pipeline(BasicPreProcessing(), CountVectorizer(ngram_range=(1,3)), MultinomialNB(alpha=0.1))
Best_SVM = make_pipeline(StemmerProcess(), TfidfVectorizer(ngram_range=(1,2)), LinearSVC(C=1, penalty = "l2", fit_intercept=False, dual=False, loss="squared_hinge", class_weight="balanced", max_iter=2000, verbose=1))
Best_Forest = make_pipeline(StemmerProcess(), CountVectorizer(ngram_range=(1,1)), RandomForestClassifier(min_samples_split = int(len(X_train)*0.01),n_estimators=500, max_features="log2", class_weight="balanced", verbose=1, n_jobs=-1))
Best_models = [Best_LR, Best_MNB, Best_SVM, Best_Forest]

for model in Best_models:
    model.fit(X_train, y_train)

def ensemble_predictions(predictions):
    #predictions is an 4 x len(train) array 
    preds = []
    for i in range(len(predictions[0])):
        pred = 0
        for model in range(4):
            if model == 0: #logistic regression
                pred += predictions[model][i] *1.3
            elif model == 4: #random forest
                pred += predictions[model][i] *0.5
            else: #svm or MNB
                pred += predictions[model][i] *1.10
        if pred / 4 > 0.55: #threshold for class classification
            preds.append(1)
        else:
            preds.append(0)
    return preds
#ensemble_preds = ensemble_predictions(predictions_all_models)

df = pd.read_csv("../input/quora-insincere-questions-classification/test.csv")
X_submission = np.array(df["question_text"])

#Ensumble model submissions
predictions_submissions = []
for model in Best_models:
    predictions_submissions.append(model.predict(X_submission))
ensemble_submissions = ensemble_predictions(predictions_submissions)

submission = pd.DataFrame.from_dict({'qid' : df['qid']})
submission['prediction'] = ensemble_submissions
submission.to_csv('submission.csv', index=False)