import numpy as numpy
from sklearn.datasets import load_iris
from sklearn import tree

# https://en.wikipedia.org/wiki/Iris_flower_data_set
iris = load_iris()

print 'List of features: ' + ' '.join(iris.feature_names)
print 'List of Labels: ' + ' '.join(iris.target_names)

print 'Sample features: ', iris.data[0]
print 'Sample Label: ', iris.target[0]

print 'Number of records: ', len(iris.data)

# Extract 3 records as testing data
# record at 0, 50, 100 index are three different labels
test_data_flower1 = iris.data[0]
test_label_flower1 = iris.target[0]

test_data_flower2 = iris.data[50]
test_label_flower2 = iris.target[50]

test_data_flower3 = iris.data[100]
test_label_flower3 = iris.target[100]

# Remove test data and label from main data to create training data
test_ids = [0, 50, 100]
training_data = numpy.delete(iris.data, test_ids, axis=0)
training_labels = numpy.delete(iris.target, test_ids)

print 'training data length: ', len(training_data)

## start training
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(training_data, training_labels)

print 'Testing ', test_data_flower1, ' for label ', test_label_flower1
print 'classification: ', classifier.predict(test_data_flower1)

print 'Testing ', test_data_flower2, ' for label ', test_label_flower2
print 'classification: ', classifier.predict(test_data_flower2)

print 'Testing ', test_data_flower3, ' for label ', test_label_flower3
print 'classification: ', classifier.predict(test_data_flower3)
