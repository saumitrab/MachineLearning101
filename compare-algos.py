import numpy as numpy
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

from sklearn import tree
classifier = tree.DecisionTreeClassifier()

classifier.fit(X_train, y_train)

preditions = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, preditions)
