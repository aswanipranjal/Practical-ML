# to be run on Anaconda
import tensorflow.contrib.learn as skflow
from sklearn import datasets, metrics

iris = datasets.load_iris()
feature_columns = skflow.infer_real_valued_columns_from_input(iris.data)
clf = skflow.LinearClassifier(n_classes=3, feature_columns=feature_columns)
clf.fit(iris.data, iris.target, steps=200, batch_size=32)
iris_predictions = list(clf.predict(iris.data, as_iterable=True))
score = metrics.accuracy_score(iris.target, clf.predict(iris.data))
print("Accuracy: %f" %score)