#!/usr/bin/env python
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def extract_features(file):
    #, name: str
    #features: Dict[str, Any] = {}
    df = pd.read_csv(file)
    df["num_sentiment"] = df.sentiment.apply(lambda x: 1 if \
    x == "positive"  else 0)
    return(df)


    def train_test_split(self,df):
        X = np.array(self.df["review"])
        y = np.array(self.df["sentiment"])
        X_train, X_test, y_train, y_test = train_test_split(X, y,  \
        test_size = .25, random_state = 42)
        return(X_train,X_test,y_train,y_test)


    def score(y_true,y_pred):
        accuracy = accuracy_score(y_true,y_pred)
        return(accuracy)
