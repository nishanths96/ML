import pandas as pd
import quandl
import math
import numpy as np #learn
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
#preprocessing is for scaling ( done on features )
import matplotlib.pyplot as plt
from matplotlib import style
import datetime
import pickle
#use Wiki dataset and take out only those features that are desired to predict the output
dataFrame = quandl.get('WIKI/GOOGL')

#we dont really require all these features.
dataFrame = dataFrame[['Adj. Open','Adj. High','Adj. Low','Adj. Close', 'Adj. Volume']] 
dataFrame['HighLowPercent'] = (dataFrame['Adj. High']-dataFrame['Adj. Low'])/dataFrame['Adj. Low'] * 100.0
dataFrame['PercentChange'] = (dataFrame['Adj. Close']-dataFrame['Adj. Open'])/dataFrame['Adj. Open'] * 100.0

# creating new dataframe with only required features.
dataFrame = dataFrame[['Adj. Close','HighLowPercent','PercentChange','Adj. Volume']]

# we need a label now ( for prediction ) 
forecastColumn = 'Adj. Close'

# rather than getting rid of data one can use this approach
dataFrame.fillna(-99999, inplace=True)

#prediction of Adj. Close for 4 days(approx. ) from the current day
forecastOut = int(math.ceil(0.1*len(dataFrame)))
dataFrame['label'] = dataFrame[forecastColumn].shift(-forecastOut)
#learn

#X -> features & y-> Label

X = np.array(dataFrame.drop(['label', 'Adj. Close'],1))
X = preprocessing.scale(X)
X_toPredict = X[-forecastOut:]
X = X[:-forecastOut]
#print(dataFrame.tail())
dataFrame.dropna(inplace=True) 
y = np.array(dataFrame['label'])
# make sure both X and Y are of the same shape
#print(X.shape, y.shape)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
predictor = LinearRegression(n_jobs=-1)		#run as many jobs as my processor can take
predictor.fit(X_train, y_train)	#fit for train

with open('linearregression.pickle', 'wb') as f:
	pickle.dump(predictor, f)
pickleinput = open('linearregression.pickle', 'rb')
predictor = pickle.load(pickleinput)

#save the model if required ( use pickle )

accuracy = predictor.score(X_test, y_test)		#score for train
forecastSet = predictor.predict(X_toPredict)
print( "Number of values to Predict: "+str(forecastOut), "PredictedValues: "+str(forecastSet), "Accuracy: "+str(accuracy))

#graphs and styling: 
style.use('ggplot')
dataFrame['Forecast'] = np.nan

lastDate = dataFrame.iloc[-1].name
lastTimeStamp = lastDate.timestamp()
oneDaySeconds = 86400
nextTimeStamp = lastTimeStamp + oneDaySeconds
for i in forecastSet:
	nextDate = datetime.datetime.fromtimestamp(nextTimeStamp)
	nextTimeStamp = nextTimeStamp+oneDaySeconds
	dataFrame.loc[nextDate] = [np.nan for x in range(len(dataFrame.columns)-1)] + [i]
dataFrame['Adj. Close'].plot()
dataFrame['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.savefig('something.png') 

