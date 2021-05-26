#!/usr/bin/env python3
import argparse
from collections import defaultdict
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
from nltk.corpus import wordnet as wn
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from sklearn import naive_bayes,svm
from sklearn.svm import SVC
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List

"""Naïve Bayes and Support vector machine classifiers for sentiment analysis. """


np.random.seed(500)

movie_sent = pd.read_csv("data/IMDB_data.csv")







"""
Index(['review', 'sentiment'], dtype='object')
"""

movie_sent["num_sent"] = movie_sent["sentiment"].apply(lambda x: 1 if x == \
"positive" else 0)

movie_sent.dropna(inplace=True)

movie_sent.review = [entry.lower() for entry in movie_sent.review]

X_train, X_test, y_train, y_test = train_test_split(movie_sent.review \
,movie_sent.num_sent,test_size=0.3,random_state=1)


tfidf_vect = TfidfVectorizer(max_features=5000)
tfidf_vect.fit(movie_sent.review)
train_x_tfidf = tfidf_vect.transform(X_train)
test_x_tfidf = tfidf_vect.transform(X_test)

def multinomial_NB():
    nb_classifier = naive_bayes.MultinomialNB()
    nb_classifier.fit(train_x_tfidf,y_train)
    predictions_nb = nb_classifier.predict(test_x_tfidf)
    print(f"multinomial NB classification accuracy {np.round(accuracy_score(predictions_nb,y_test),2)*100}")
multinomial_NB()



def svm():
    sv_classifier = SVC(kernel='linear')
    sv_classifier.fit(train_x_tfidf,y_train)
    predictions_sv = sv_classifier.predict(test_x_tfidf)
    print(f"svm acuracy {accuracy_score(predictions_sv,y_test)*100}")
svm()



def random_forest():
    rf_classifier = RandomForestClassifier(n_estimators=100)
    rf_classifier.fit(train_x_tfidf,y_train)
    predictions_RF = RF_classifier.predict(test_x_tfidf)
    print(f"RF acuracy {np.round(accuracy_score(predictions_RF,y_test)*100,decimals=2)}")
random_forest()



def LogisticRegression():
    lr_model = sklearn.linear_model.LogisticRegression(
        penalty="l1",
        C=100,
        solver="liblinear"
    )
    lr_model.fit(train_x_tfidf,y_train)
    predictions_LR = lr_model.predict(test_x_tfidf)
    print(f"logistic regression acuracy {np.round(accuracy_score(predictions_LR,y_test)*100,decimals=2)}")
LogisticRegression()





"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("train", help="path to input train TSV")
    parser.add_argument("lr_model", help="path to output model")
    main(parser.parse_args())
"""
