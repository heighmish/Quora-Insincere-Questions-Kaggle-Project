# Quora Insincere Questions Readme

There are 5 python files in this directory that were used to solve the Quora insincere question problem. All files require the data is stored in the data directory such as
"./Data/train.csv".

1: HelperModule.py
This file includes a function (read_data) that takes a filename, reads in a takes only 10.5% of the data returns two numpy arrays, features and target values.
The file includes 3 helper classes that extend Sklearn estimators and transformer classes to be compatible with Sklearn pipelines. 
	1: modelSwitcher: class that can accept any valid sklearn model, LogisticRegression, RandomForestClassifier() etc.
	2: StemmerProcess: transformer class that applies default word transformation NLTK porter stemming to a dataset
	3: BasicPreProcessing: transformer class that applies default word transformation no porter stemming

2: ModelSearch.py
File contains all the parameters and gridsearches used to generate the data for the report. Uses pipelines made up of the classes from HelperModule.py.
File takes a very long time (~10 hours on my machine) to run as there are many expensive grid searches being run. Unfortunately, the script was run before the announcement
to post cached results and I did not save the results at the time. To get the best model for each gridsearch I looked through the cv_results_["mean_test_score"] and found
the best score for each model.

3: BestModel.py
File that has the best pipeline version of each model found by ModelSearch.py. File fits models to training data then generates prediction results on the independent test set
to use to create various graphs and other useful measures. File includes the method for ensembling the models and generating the predictive label new tests. For some graphs in 
the report, the data was taken from the python standard output and graphs were customed designed in Excel. 

4: addition_experiments
File contains an additional experiement related to feature selection. The results of that experiement are at the end of the appendix.

5: Submitted.py
Contains the exact notebook file from kaggle used for prediction. It is a combination of the methods from BestModel.py and HelperModule.py. The other major difference is the 
read_data function reads all rows of the dataframe rather than 10.5% and includes a method at the bottom to predict on the test set and output the predictions to csv format.
This file will not run due to trying to read in directories exclusive to the kaggle virtual environment.
Link to the public notebook is below.

https://www.kaggle.com/neighmish/fork-of-quora-insincere-notebook?scriptVersionId=69453681


