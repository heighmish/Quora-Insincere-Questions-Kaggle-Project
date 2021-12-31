from HelperModule import *

"""
File calculates the correlation between both average word length, and median word length in a given sentance, with the target class value
"""
def make_median(row):
    temp = [len(i) for i in row.split(" ")]
    return np.median(temp)

def make_mean(row):
    temp = [len(i) for i in row.split(" ")]
    return np.mean(temp)

df = pd.read_csv("./Data/train.csv")

def word_length(df):
    df["med_length"] = df["question_text"].apply(make_median)
    df["mean_length"] = df["question_text"].apply(make_mean)
    data = list(["target", "med_length", "mean_length"])
    correlation = df[data].corr()
    print(correlation["target"])


word_length(df)

