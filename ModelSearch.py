#File contains all relevant grid searches
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from HelperModule import *

X,y = read_data("./Data/train.csv", 0.105)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state=0)
print(len(X_train), len(X_test))

#models are named after the naming convention described in the report

#Baseline models
print("A STARTED")
A_params = [
    {
        'modelswitcher__estimator': [LogisticRegression(class_weight="balanced", solver="liblinear")],
    },
    {
        'modelswitcher__estimator': [MultinomialNB()],
    },
    {
        'modelswitcher__estimator': [LinearSVC(fit_intercept=False, loss="squared_hinge", class_weight="balanced")],
    },
    {
        'modelswitcher__estimator': [RandomForestClassifier(n_estimators=25, max_features="log2", min_samples_split = int(len(X_train)*0.01), class_weight="balanced")],
    }
]
A_pipeline = make_pipeline(BasicPreProcessing(), CountVectorizer(), modelSwitcher())
AGS = GridSearchCV(A_pipeline, A_params, cv=3, verbose=3, n_jobs=-1,  scoring="f1") 
AGS.fit(X_train, y_train)
print(f"Best CV score: {AGS.best_score_}")
print(f"Best CV params: {AGS.best_params_}")
print([AGS.cv_results_["mean_test_score"], AGS.cv_results_["std_test_score"]])


#no model hypertuning only text preprocessing methods
print("B STARTED")
B_pipeline = make_pipeline(StemmerProcess(), CountVectorizer(stop_words="english", lowercase=True, strip_accents="unicode"), modelSwitcher())
BGS = GridSearchCV(B_pipeline, A_params, cv=3, verbose=3, n_jobs=-1,  scoring="f1") 
BGS.fit(X_train, y_train)
print(f"Best CV score: {BGS.best_score_}")
print(f"Best CV params: {BGS.best_params_}")
print(BGS.cv_results_["mean_test_score"])


C_params = [
    {
        'modelswitcher__estimator': [LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=300)],
        'modelswitcher__estimator__C' : [0.01, 0.1, 1, 10, 100],
        'modelswitcher__estimator__penalty' : ('l1', 'l2'),
        'countvectorizer__stop_words' : ("english", None),
        'countvectorizer__ngram_range' : [(1,1), (1,2), (1,3), (1,4)]
    },
    {
        'modelswitcher__estimator': [MultinomialNB()],
        'modelswitcher__estimator__alpha' : [0.01, 0.1, 1, 10, 100],
        'countvectorizer__stop_words' : ("english", None),
        'countvectorizer__ngram_range' : [(1,1), (1,2), (1,3), (2,2), (1,4)]
        
    },
    {
        'modelswitcher__estimator': [LinearSVC(fit_intercept=False, dual=False, loss="squared_hinge", class_weight="balanced", max_iter=2000)],
        'modelswitcher__estimator__penalty' : ('l1', 'l2'),
        'modelswitcher__estimator__C' : [0.001, 0.01, 0.1, 1, 10, 100],
        'countvectorizer__stop_words' : ("english", None),
        'countvectorizer__ngram_range' : [(1,1), (1,2), (1,3)]
        
    },
    {
        'modelswitcher__estimator': [RandomForestClassifier(max_features="log2", class_weight="balanced")],
        'modelswitcher__estimator__n_estimators': [10, 25, 50, 100, 250, 500],
        'modelswitcher__estimator__min_samples_split': [int(len(X_train)*0.1), int(len(X_train)*0.005), int(len(X_train)*0.001)],
        'countvectorizer__stop_words' : ("english", None),
        'countvectorizer__ngram_range' : [(1,1), (1,2)]
    }

]

#Grid search with stemming, count vectorizer
print("C1 STARTED")
C_pipeline_stem = make_pipeline(StemmerProcess(), CountVectorizer(), modelSwitcher())
CGS_stem = GridSearchCV(C_pipeline_stem, C_params, cv=3, verbose=3, n_jobs=-1,  scoring="f1") 
CGS_stem.fit(X_train, y_train)
print(f"Best CV score: {CGS_stem.best_score_}")
print(f"Best CV params: {CGS_stem.best_params_}")
print(CGS_stem.cv_results_["mean_test_score"])

#Grid search without stemming, count vectorizer
print("C2 STARTED")
C_pipeline_not_stem = make_pipeline(BasicPreProcessing(), CountVectorizer(), modelSwitcher())
CGS_not_stem = GridSearchCV(C_pipeline_not_stem, C_params, cv=3, verbose=3, n_jobs=-1,  scoring="f1") 
CGS_not_stem.fit(X_train, y_train)
print(CGS_not_stem.cv_results_)

D_params = [
    {
        'modelswitcher__estimator': [LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=300)],
        'modelswitcher__estimator__C' : [0.01, 0.1, 1, 10, 100],
        'modelswitcher__estimator__penalty' : ("l1", "l2"),
        'tfidfvectorizer__stop_words' : ("english", None),
        'tfidfvectorizer__ngram_range' : [(1,1), (1,2), (1,3), (1,4)]
    },
    {
        'modelswitcher__estimator': [MultinomialNB()],
        'modelswitcher__estimator__alpha' : [0.01, 0.1, 1, 10, 100],
        'tfidfvectorizer__stop_words' : ("english", None),
        'tfidfvectorizer__ngram_range' : [(1,1), (1,2), (1,3), (2,2), (1,4)]
        
        
    },
    {
        'modelswitcher__estimator': [LinearSVC(fit_intercept=False, dual=False, loss="squared_hinge", class_weight="balanced", max_iter=2000)],
        'modelswitcher__estimator__C' : [0.001, 0.01, 0.1, 1, 10, 100],
        'modelswitcher__estimator__penalty' : ("l1", "l2"),
        'tfidfvectorizer__stop_words' : ("english", None),
        'tfidfvectorizer__ngram_range' : [(1,1), (1,2), (1,3)]
        
    },
    {
        'modelswitcher__estimator': [RandomForestClassifier(max_features="log2", class_weight="balanced")],
        'modelswitcher__estimator__n_estimators': [10, 25, 50, 100, 250, 500],
        'modelswitcher__estimator__min_samples_split': [int(len(X_train)*0.1), int(len(X_train)*0.005), int(len(X_train)*0.001)],
        'tfidfvectorizer__stop_words' : ("english", None),
        'tfidfvectorizer__ngram_range' : [(1,1), (1,2)]
    }

]
#Grid search with stemming, Tfidfvectorizer
print("D1 STARTED")
D_pipeline_stem = make_pipeline(StemmerProcess(), TfidfVectorizer(), modelSwitcher())
DGS_stem = GridSearchCV(D_pipeline_stem, D_params, cv=3, verbose=3, n_jobs=-1,  scoring="f1") 
DGS_stem.fit(X_train, y_train)
print(f"Best CV score: {DGS_stem.best_score_}")
print(f"Best CV params: {DGS_stem.best_params_}")
print(DGS_stem.cv_results_["mean_test_score"])

#Grid search without stemming, Tfidfvectorizer
print("D2 STARTED")
D_pipeline_not_stem = make_pipeline(BasicPreProcessing(), TfidfVectorizer(), modelSwitcher())
DGS_not_stem = GridSearchCV(D_pipeline_not_stem, D_params, cv=3, verbose=3, n_jobs=-1,  scoring="f1") 
DGS_not_stem.fit(X_train, y_train)
print(DGS_not_stem.cv_results_)