import pickle, gzip
from sklearn import svm, preprocessing, cross_validation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

with gzip.open('mnist.pkl.gz', 'rb') as f:
	train_set, valid_set, test_set=pickle.load(f, encoding='latin1')
X = np.array(train_set[:-1][0])
y = np.array(train_set[-1:][0])

classifier = svm.SVC(decision_function_shape='ovr')
trainX, testX, train_y, test_y  = cross_validation.train_test_split(X, y, test_size=0.2)
classifier.fit(trainX, train_y)
with open('svmmnist.pickle', 'wb') as f:
	pickle.dump(classifier, f)
accuracy = classifier.score(testX, test_y)
print(accuracy)
input_given = np.array(test_set[:-1][0])
output_expected = np.array(test_set[-1:][0])
newin = input_given.reshape(len(input_given) ,-1)
print(len(valid_set))
with open('svmmnist.pickle', 'rb') as f:
	classifier = pickle.load(f)
	
	o = classifier.predict(newin)
	print(o)

# obtained accuracy 93 percent

