from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.porter import *
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

from HelperModule import *

X,y = read_data("./Data/train.csv", 0.105)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state=0)
print(len(X_train), len(X_test))


#These are the best models found using ModelSearch.py
Best_LR = make_pipeline(StemmerProcess(), CountVectorizer(ngram_range=(1,3)), LogisticRegression(C=1, solver="liblinear", class_weight="balanced",penalty="l2", max_iter=200, verbose=1, n_jobs=-1))
Best_MNB = make_pipeline(BasicPreProcessing(), CountVectorizer(ngram_range=(1,3)), MultinomialNB(alpha=0.1))
Best_SVM = make_pipeline(StemmerProcess(), TfidfVectorizer(ngram_range=(1,2)), LinearSVC(C=1, penalty = "l2", fit_intercept=False, dual=False, loss="squared_hinge", class_weight="balanced", max_iter=2000, verbose=1))
Best_Forest = make_pipeline(StemmerProcess(), CountVectorizer(ngram_range=(1,1)), RandomForestClassifier(min_samples_split = int(len(X_train)*0.01),n_estimators=500, max_features="log2", class_weight="balanced", verbose=1, n_jobs=-1))
Best_models = [Best_LR, Best_MNB, Best_SVM, Best_Forest]


#predict on test set and save results for use in report
reports = [] # used for the two bar graphs, appendix 3B
scores = [] # used for the two bar graphs, appendix 3B
Names = ["Logistic Regression", "Multinomial Naive Bayes", "Support Vector Machine", "Random Forest"]
for model in Best_models:
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    reports.append(classification_report(y_test, predictions))
    scores.append(f1_score(y_test, predictions))

for report in reports:
    print(report)
    
    
#Generate test set predictions of each of the models 
predictions_all_models = []
for model in Best_models:
    predictions_all_models.append(model.predict(X_test))
    
def ensemble_threshold_graph(predictions): 
    f1_scores_thresholds = []
    thresholds = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55,0.6,0.65,0.70,0.75,0.8,0.85,0.9,0.95,1]
    for threshold in thresholds:
        preds = []
        for i in range(len(predictions[0])):
            pred = 0
            for j in range(4):
                if j == 0:
                    pred += predictions[j][i] *1.3
                elif j == 4:
                    pred += predictions[j][i] *.5
                else:
                    pred += predictions[j][i] *1.10
            if pred / 4 > threshold:
                preds.append(1)
            else:
                preds.append(0)
        f1_scores_thresholds.append(f1_score(y_test, preds))
    print(max(f1_scores_thresholds), thresholds[f1_scores_thresholds.index(max(f1_scores_thresholds))])
    plt.plot(thresholds, f1_scores_thresholds)
    plt.title("F1_score for Various Thresholds")
    plt.ylabel("F1 Score")
    plt.xlabel("Threshold values")
    plt.savefig(fname="ensumbleThresholds.png")
    plt.show()
ensemble_threshold_graph(predictions_all_models)


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
ensumble_preds = ensemble_predictions(predictions_all_models)

print(classification_report(y_test, ensumble_preds))
#Recall has increased significantly


def create_important_features(): #create the graph of features that have the highest and lowest coefficent score for the best logistic regression model
    named_coefficients = list(zip((Best_LR.named_steps["countvectorizer"]).get_feature_names(), (Best_LR.named_steps["logisticregression"]).coef_[0]))
    named_coefficients = sorted(named_coefficients, key = lambda x:x[1])
    highest = []
    lowest = []
    for i in range(26,1, -1):
        highest.append(named_coefficients[-i])
    for i in range(1, 26):
        lowest.append(named_coefficients[i])

    plt.figure(figsize=(16,5))
    plt.bar(*zip(*lowest), width=.85, ec='black', color = "#91C4F2")
    plt.bar(*zip(*highest), width=.85, ec='black', color = "#A14DA0")
    plt.title("Logistic Regression Top 50 Highest and Lowest Coeficients Values (Higher Coefficient Implies Insincere Question)")
    plt.xlabel("Feature")
    plt.ylabel("Coefficient Value")
    plt.xticks(rotation=80, fontsize=12.5)
    plt.xlim([-1, 50])
    plt.savefig(fname="LR Top and bottom coefficients", bbox_inches='tight', dpi=480, facecolor='w')
    plt.show()
create_important_features()