import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib import style 
from collections import Counter
import pandas as pd
import random
style.use('fivethirtyeight')
dataset = {'k': [[1,2], [2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
new_features = [5,7]

#visualization: 
[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=100, color='g')
plt.savefig('dataset.png')

def k_nearest_neighbors(data, predict, k=3):
	#print(len(data))
	if len(data) >=k:
		warning.warn('k is less than total types of groups')
	distances = []
	for group in data:
		for features in data[group]:
			euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidean_distance, group])
	votes = [i[1] for i in sorted(distances)[:k]]
	#print(votes)
	#print(Counter(votes).most_common(1)[0])
	#print(Counter(votes).most_common(1)[0][0])
	vote_result = Counter(votes).most_common(1)[0][0]
	return vote_result

result = k_nearest_neighbors(dataset, new_features, k=3)
#print(result)
[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=100, color=result)
plt.savefig('result.png')

'''
dataFrame = pd.read_csv('breast-cancer-wisconsin.data.txt')
dataFrame.replace('?', -99999, inplace=True)
df.drop(['id'], axis=1, inplace=True)
#converting everything to float just to makesure that everything is of the same type!
full_data = dataFrame.astype(float).values.tolist()
#shuffle the data
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = full_data[: -int(test_size*len(full_data))]
test_data = full_data[-int(test_sizealen(full_data)):]

for i in train_data:
	train_set[i[-1]].append(i[:-1])
#continues...
'''
