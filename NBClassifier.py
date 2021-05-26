#!/usr/bin/env python
"""MultinomialNB classifier for pet names."""



from typing import Any, Dict, List
import numpy as np
import pandas as pd
import sklearn.feature_extraction  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn  # type: ignore
import utils


class NBClassifier:
    """This class stores code for extracting features,
    training, and predicting, along with associated model
    and vectorization data."""

    def __init__(self):
        #self.tf_vectorizer = TfidfVectorizer()
        self.cnt_vectorizer = CountVectorizer()

        # The term "Bernoulli" here refers to Bernoulli trials, observations
        # that can either be true or false. Here these correspond to the
        # vectorizer's binarization of the features.
        self.classifier = MultinomialNB()

#X_train = vectorizer.fit_transform(X_train)
#X_test = vectorizer.transform(X_test)



    def train(self, xx, y):
        xx = self.cnt_vectorizer.fit_transform(
            np.array(xx)
        )
        self.classifier.fit(xx, y)

    def predict(self, x: List[str]) -> List[str]:
        xx = self.cnt_vectorizer.transform(
            np.array(x)
            )
        return list(self.classifier.predict(xx))
"""
def main():
    model = MNNaiveBayesClassifier()
    data = utils.extract_features("data/IMDB_data.csv")
    X_train,X_test,y_train,y_test = utils.train_test_split(data["review"],data["num_sentiment"])
    model.train(X_train,y_train)
    model.predict(X_test)

main()
"""
