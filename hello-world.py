from sklearn import tree

# weight isSmooth label
# 140    1        0     - Apple
# 130    1        0     - Apple
# 150    0        1     - Orange
# 170    0        1     - Orange


features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels)

# predict weight 160, isSmooth 0
print classifier.predict([[160, 0]])
