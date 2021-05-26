#!/usr/bin/env/python
import argparse
import ReviewLogClassifier
#import NBClassifier
#import RandomForestClassifier
#import SupportVectorClassifier
import utils
import sklearn
from sklearn.metrics import accuracy_score


#import file name.
#file_name.class_name_classifier is an instance of the model

"""
df = utils.extract_features("data/IMDB_data.csv")


X_train, X_test, y_train, y_test = utils.train_test_split(df["review"],df["num_sentiment"])
#print(X_train)

#print(X_train)
lr_model = LR_class.ReviewLogClassifier()
lr_model.train(X_train,y_train)
preds = lr_model.predict(X_test)
score = accuracy_score(preds,y_test)
print("Logistic_regression: ",score)


nb_model = multinomial_NB.NBClassifier()
nb_model.train(X_train,y_train)
preds = nb_model.predict(X_test)
score = accuracy_score(preds,y_test)
print("multinomial nb: ", score)



rf_model = random_forest_classifier.RandomForestClassifier()
rf_model.train(X_train,y_train)
preds = rf_model.predict(X_test)
score = accuracy_score(preds,y_test)
print("random forest: ", score)


svm_model = support_vector_classifier.SupportVectorClassifier()
svm_model.train(X_train,y_train)
preds = svm_model.predict(X_test)
score = accuracy_score(preds,y_test)
print("support vector score: ", score)

"""

def main(args: argparse.Namespace)-> None:
    classifier = args.model
    classifier.train(X_train,y_train)
    #preds = classifier.predict(X_test)
    #score = accuracy_score(preds,y_test)
    #print(f"model score: {score}")
    #print(type(classifier))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model",help="specify the model to use")
    main(parser.parse_args())

#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description=__doc__)
    #parser.add_argument("test", help="path to input test TSV")
#    parser.add_argument("model", help="path to input model")
#    main(parser.parse_args())
