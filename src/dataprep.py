import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def prepare_data():
    mushroom = pd.read_csv(os.getcwd() + '/mushrooms.csv')
    X = mushroom.copy()  # create a deep copy of the dataset
    # transform class labels to numbers
    X['class'] = X['class'].map({'p': 1, 'e': 0})

    features_cat = {
        "cap-shape": ('b', 'c', 'x', 'f', 'k', 's'),
        "cap-surface": ('f', 'g', 'y', 's'),
        "cap-color": ('n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'),
        "bruises": ('t', 'f'),
        "odor": ('a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'),
        "gill-attachment": ('a', 'd', 'f', 'n'),
        "gill-spacing": ('c', 'w', 'd'),
        "gill-size": ('b', 'n'),
        "gill-color": ('k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'),
        "stalk-shape": ('e', 't'),
        "stalk-root": ('b', 'c', 'u', 'e', 'z', 'r', '?'),
        "stalk-surface-above-ring": ('f', 'y', 'k', 's'),
        "stalk-surface-below-ring": ('f', 'y', 'k', 's'),
        "stalk-color-above-ring": ('n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'),
        "stalk-color-below-ring": ('n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y'),
        "veil-type": ('p', 'u'),
        "veil-color": ('n', 'o', 'w', 'y'),
        "ring-number": ('n', 'o', 't'),
        "ring-type": ('c', 'e', 'f', 'l', 'n', 'p', 's', 'z'),
        "spore-print-color": ('k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y'),
        "population": ('a', 'c', 'n', 's', 'v', 'y'),
        "habitat": ('g', 'l', 'm', 'p', 'u', 'w', 'd'),
    }

    labelencoder = LabelEncoder()
    for key in features_cat:
        # reassign columns of encoded labels in number
        X[key] = labelencoder.fit_transform(X[key])

    # training set is the 70% of the data
    df_train = X.sample(frac=0.7, random_state=0)

    # ignore the class column in features
    X_train = df_train.drop('class', axis=1)
    X_valid = X.drop('class', axis=1)
    # targets
    y_train = df_train['class']
    y_valid = X['class']

    return X_train, y_train, X_valid, y_valid, X_train.shape[1]
