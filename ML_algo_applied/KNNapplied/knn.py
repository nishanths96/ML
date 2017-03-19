import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd
import pickle
dataFrame = pd.read_csv('breast-cancer-wisconsin.data.txt')
dataFrame.replace('?', -99999, inplace=True)
#id doesn't influence the output...not required
dataFrame.drop(['id'], axis=1, inplace=True)
 
X =  np.array(dataFrame.drop(['class'], axis=1))
y = np.array(dataFrame['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

classifier = neighbors.KNeighborsClassifier()
# incase of SVM and polynomial kernel use:
#classifier = svm.SVC(kernel='poly')
classifier.fit(X_train, y_train)
with open('knn.pickle','wb') as f:
	pickle.dump(classifier, f)
picklein = open('knn.pickle','rb')
classifier = pickle.load(picklein)
accuracy = classifier.score(X_test, y_test)
print(accuracy)

exampleTotest = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])
#exampleTotest = np.array([[4,2,1,1,1,2,3,2,1], [4,2,3,4,6,8,5,4,5], [4,8,9,4,6,8,6,7,5]]) 
#len(exampleTotest) = 2
exampleTotest = exampleTotest.reshape(len(exampleTotest), -1)
predictor = classifier.predict(exampleTotest)
print(predictor)
