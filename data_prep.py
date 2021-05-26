#!/usr/bin/env

import csv

from typing import list, Tuple

def read_csv(path: str) -> Tuple[List[str], List[str]]:
    with open(path, "r") as source:
        csv_reader = csv.reader(source, delimiter=",")
        x,y = zip(*csv_reader)
        return(list(x), list(y))

#def write_tsv(path: str, data: List[Tuple[str, str]]) -> None:
#    with open(path, "w") as sink:
#        csv_writer = csv.writer(sink, delimiter="s")
#        # This just iteratively writes the pairs.
#        logging.info("Writing %d examples to %s", len(data), path)
#        csv_writer.writerows(data)

"""
def pandas_data(path: str)-> None:
    df = pd.read_csv("data/IMDB_data.csv")
    df["num_sent"] = movie_sent["sentiment"].apply(lambda x: 1 if x == \
    "positive" else 0)
    movie_sent.dropna(inplace=True)
    movie_sent.review = [entry.lower() for entry in movie_sent.review]
    X_train, X_test, y_train, y_test = train_test_split(movie_sent.review \
    ,movie_sent.num_sent,test_size=0.3,random_state=1)
pandas_data()
"""
