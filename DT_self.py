import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import train_test_split


class DTClassifier:
    root = None

    def __init__(self):
        print("")

    def predict(self, x_test):
        print("\nIs", self.root, "?")
        print(" --> Has Lung Cancer?")
        return x_test[self.root]

    def calculate_gini(self, a, b):
        p = (a + b)
        gini = 1 - (a / p) ** 2 - (b / p) ** 2
        return gini

    def train(self, x, y):
        print("   True     False")
        a = x.loc[(x == 2) & (y == "2")].count()
        b = x.loc[(x == 2) & (y == "1")].count()
        c = x.loc[(x == 1) & (y == "2")].count()
        d = x.loc[(x == 1) & (y == "1")].count()

        print(" ", a, b, " ", c, d, "\n")
        impurity_true = self.calculate_gini(a, b)
        print("Gini True:", impurity_true)

        impurity_false = self.calculate_gini(c, d)
        print("Gini False:", impurity_false)

        average_gini = ((a + b) / (a + b + c + d)) * impurity_true + ((c + d) / (a + b + c + d)) * impurity_false
        print("Average Gini:", average_gini)

        return average_gini

    def fit(self, x_train, y_train):
        self.x = x_train
        self.y = y_train

        best_gini = None
        tmp = 1
        for column_names in self.x:
            print("\n    ", column_names + "?")
            gini = self.train(self.x[column_names], self.y)
            if gini < tmp:
                best_gini = column_names
                tmp = gini
        print("\nFirst node of the decision tree is:", best_gini)
        self.root = best_gini


def main():
    df = pd.read_csv("survey lung cancer.csv")

    # Create targets
    y = df.pop("LUNG_CANCER")
    y = y.replace({"NO": "1", "YES": "2"})

    # Create features
    x = df[["SMOKING", "CHEST PAIN", "ALCOHOL CONSUMING", "PEER_PRESSURE", "CHRONIC DISEASE", "ANXIETY"]]

    # Split dataset in random training and testing datasets
    x_train, x_test, y_train, y_test = train_test_split(x, y)

    # Create classifier
    clf = DTClassifier()

    # Train model
    clf.fit(x_train, y_train)

    # Predict
    prediction = clf.predict(x_test)

    counter = 0
    accuracy = 0
    y_test = y_test.tolist()

    for x in prediction:
        color_s = "\033[92m"
        color_e = "\033[0m"
        if int(y_test[counter]) == int(x):
            comparison = True
            accuracy += 1
        else:
            comparison = False
            color_s = "\033[91m"

        print("  expect:", y_test[counter], "  predict:", x, " ", color_s,
              comparison, color_e)

        counter += 1
    print("The first question predicted the correct target with a probability of", accuracy / prediction.count() * 100,
          "%.")


if __name__ == '__main__':
    main()
