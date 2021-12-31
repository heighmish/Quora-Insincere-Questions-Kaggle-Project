#pip instal nltk 
#nltk.download('punkt') #nltk requires additional downloads for certain functionality

from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.porter import *
from nltk.tokenize import word_tokenize
from string import punctuation
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from string import punctuation



#Class inspired by https://stackoverflow.com/questions/50285973/pipeline-multiple-classifiers
#Class accepts an sklearn estimator, can be used in gridsearch pipelines to test multiple different models
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
    
def read_data(filename, percent=1):
    """
    Takes in a filename returns features and targets numpy array
    """
    df = pd.read_csv(filename)
    print(np.mean(df["target"]))
    train = df[:int(len(df)*percent)]
    print(np.mean(train["target"]))
    X = np.array(train["question_text"])
    y = np.array(train["target"])
    return X, y





def remove_punctuation(word):
    return ''.join(c for c in word if c not in punctuation)
    

class BasicPreProcessing(BaseEstimator, TransformerMixin):
    """
    custom class that can be used in a sklearn pipeline
    tokenizes words, removes punction and any words
    less than 2 characters
    """
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
        

class StemmerProcess(BaseEstimator, TransformerMixin):
    """
    custom class that can be used in a sklearn pipeline
    implements all preprocessing methods that BasicPreProcessing does
    and also applies porter stemming to each word in the document set
    """
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


