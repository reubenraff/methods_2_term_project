"""LogisticRegression classifier for pet names."""


from typing import Any, Dict, List
import numpy as np
import pandas as pd
import sklearn.feature_extraction  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn  # type: ignore
import sklearn.feature_extraction  # type: ignore
import sklearn  # type: ignore


class SupportVectorClassifier:
    """This class stores code for extracting features,
    training, and predicting, along with associated model
    and vectorization data."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        # The term "Bernoulli" here refers to Bernoulli trials, observations
        # that can either be true or false. Here these correspond to the
        # vectorizer's binarization of the features.
        self.classifier = sklearn.svm.SVC(kernel="linear")


    def train(self, x: List[str], y: List[str]):
        xx = self.vectorizer.fit_transform(x)
        self.classifier.fit(xx, y)

    def predict(self, x: List[str]) -> List[str]:
        xx = self.vectorizer.transform(x)
        return list(self.classifier.predict(xx))
