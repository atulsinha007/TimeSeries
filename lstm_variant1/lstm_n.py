#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 01:11:51 2018

@author: adi
"""
import numpy
import copy
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import re
from pandas import read_csv
import math
from keras.models import model_from_json
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
### convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return dataX, dataY




### fix random seed for reproducibility
numpy.random.seed(7)
days_ahead = 3
### load the dataset
file_st = 'Indu.csv'
price_ind = 4 	#1, 2, 3 or 4
reg_st = "(.*)\..*"
matched_obj = re.match(reg_st, file_st)
prefix_st = matched_obj.groups()[0]+'_'+str(price_ind)
dataframe = read_csv(file_st, usecols = [price_ind],  engine='python', skipfooter=0)
dataset = dataframe.values
dataset = dataset.astype('float32')
print(dataset.shape)
dataset = dataset[:-days_ahead]
print(dataset.shape)
print(type(dataset), dataset)
### normalize the dataset
#
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
data = list(dataset)
def predict_from_saved_model(trainX1):
	global prefix_st
	with open('models/model_architecture_'+prefix_st+'.json', 'r') as f:
		model = model_from_json(f.read())
	model.load_weights('models/model_weights_'+prefix_st+'.h5')
	return model.predict(trainX1)

print(len(data))
train_size = int(len(dataset) * 0.90)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

look_back = 1 ###window_size
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
futureElements=[]
nop=days_ahead 
model = Sequential()
	
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(3))

model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
with open('models/model_architecture_'+prefix_st+'.json', 'w') as f:
	f.write(model.to_json())
for i in range(nop):
	
	datalis=numpy.array(data)
	train_size = int(len(datalis) * 0.90)
	test_size = len(datalis) - train_size
	train, test = datalis[0:train_size,:], datalis[train_size:len(datalis),:]
	trainX, trainY = create_dataset(train, look_back)
	testX, testY = create_dataset(test, look_back)
	temp=copy.deepcopy(testX)
	store=copy.deepcopy(temp)
	print(len(train))
	print(len(test))
	
	a=[test[len(test)-2]]
	b=[test[len(test)-1]]
	store.append(a)
	store.append(b)
	
	trainX1=numpy.array(trainX)

	testX1=numpy.array(testX)
	trainY1=numpy.array(trainY)
	testY1=numpy.array(testY)
	store=numpy.array(store)
	
	### reshape input to be [samples, time steps, features]
	trainX1 = numpy.reshape(trainX1, (trainX1.shape[0], 1, trainX1.shape[1]))
	testX1 = numpy.reshape(testX1, (testX1.shape[0], 1, testX1.shape[1]))

	### create and fit the LSTM network
	
	model.fit(trainX1, trainY1, epochs=100, batch_size=50, verbose=2)
	### make predictions
	model.save('models/model_weights_'+prefix_st+'.h5')


	print(i)
	trainPredict = predict_from_saved_model(trainX1)
	testPredict = predict_from_saved_model(testX1)

	p=[testY1[-1]]
	q=testPredict[-1][0]

	
	data.append([q])
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY1 = scaler.inverse_transform([trainY1])
	testPredict = scaler.inverse_transform(testPredict)
	testY1 = scaler.inverse_transform([testY1])

	w=[testPredict[-1][0]]

	
	
	trainScore = math.sqrt(mean_squared_error(trainY1[0], trainPredict[:,0]))
	print_st1 = 'Train Score: %.2f RMSE\n' % (trainScore)
	print(print_st1)
	
	
	testScore = math.sqrt(mean_squared_error(testY1[0], testPredict[:,0]))
	print_st2 = 'Test Score: %.2f RMSE\n' % (testScore)
	print(print_st2)
	with open("log_folder/log_"+prefix_st+".txt", "w+") as f:
		f.write(print_st1+print_st2)
	
dataset1=numpy.array(data)
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan

trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict    
testPredictPlot = numpy.empty_like(dataset1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset1)-2, :] = testPredict

plt.plot(scaler.inverse_transform(dataset1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)

plt.savefig("graph_10.png")    
data = list(dataset)    
print(len(data))
model = Sequential()
#old_model=model
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(3))
#model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
with open('models/model_architecture_'+prefix_st+'.json', 'w') as f:
	f.write(model.to_json())

for i in range(nop):

	datalis=numpy.array(data)
	train_size = int(len(datalis) * 1.0)
	test_size = len(datalis) - train_size
	train, test = datalis[0:train_size,:], datalis[train_size:len(datalis),:]
	trainX, trainY = create_dataset(train, look_back)

	print(len(train))
	print(len(test))

	trainX1=numpy.array(trainX)

	trainY1=numpy.array(trainY)

	### reshape input to be [samples, time steps, features]
	trainX1 = numpy.reshape(trainX1, (trainX1.shape[0], 1, trainX1.shape[1]))

	
	model.fit(trainX1, trainY1, epochs=100, batch_size=50, verbose=2)
	model.save('models/model_weights_'+prefix_st+'.h5')

	print(i)
	trainPredict = predict_from_saved_model(trainX1)

	q=trainPredict[-1][0]

	
	data.append([q])
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY1 = scaler.inverse_transform([trainY1])

	futureElements.append(trainPredict[-1][0])
	
	

with open("log_folder/future_"+prefix_st+".txt", "w+") as f:
	f.write(str(futureElements)+"\n")

print(futureElements)

