import numpy as numpy
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

# split data in test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

treeClassifier = tree.DecisionTreeClassifier()
treeClassifier.fit(X_train, y_train)
treePreditions = treeClassifier.predict(X_test)

KNClassifier = KNeighborsClassifier()
KNClassifier.fit(X_train, y_train)
KNPredictions = KNClassifier.predict(X_test)

print 'Decision Tree Accuracy: ', accuracy_score(y_test, treePreditions)
print 'K Neighbors Accuracy: ', accuracy_score(y_test, KNPredictions)
