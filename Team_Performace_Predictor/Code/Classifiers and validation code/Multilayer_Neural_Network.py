import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd

#learning rate paramters
eps = 0.01
const_lambda = 0.05

#number of nodes in hidden layer
num_nodes = 7

#number of iterations
itr = [250,500,750,1000,2000,4000,5000]

np.random.seed()

#Function for derivative of tanh activation function
def derivative_tanh(x):
	return (1 - (np.power(np.tanh(x),2)))

#function for softmax
def softmax(y_cap_temp):
	temp = np.exp(y_cap_temp)
	y_cap_final = temp / np.sum(temp, axis=1, keepdims=True)
	return y_cap_final

#function for back propogation algorithm - to update the weight vectors
def neural_model(train_X, train_y,pw1,pw2,iterations):
	#running for a given number of iterations
	for j in range(0,iterations): 	    
		a1 = np.tanh(np.dot(train_X, pw1))	   #activation function to input of hidden layer
		a2 = np.dot(a1, pw2)	
		y_cap = np.tanh(a2)			#predicted class label
		delt = softmax(y_cap)		#softmax of nonlinear output

		#calculating the mean square error between predicted and actual class
		for i in range(0,train_X.shape[0]):
			delt[i] = delt[i] - train_y[i]	

		#Gradient descent algorithm
		q1 = derivative_tanh(a2) * delt             #derivative of error w.r.t w2
		dw2 = np.dot(a1.T,q1) + (const_lambda * pw2)

		temp_1 = delt * derivative_tanh(a2)			#derivative of error w.r.t w1
		temp_2 = np.dot(temp_1,pw2.T)
		temp_3 = temp_2 * derivative_tanh(np.dot(train_X, pw1))
		dw1 = np.dot(train_X.T , temp_3) + (const_lambda * pw1)	

		#updation of weight vectors
		pw2 += -eps * dw2
		pw1 += -eps * dw1
	return pw1,pw2
	
#to predict the class - forward propogation
def neural_predict(test_X,kw1,kw2):
	a1 = np.tanh(np.dot(test_X, kw1))	
	a2 = np.dot(a1, kw2)
	y_cap = np.tanh(a2)
	y_cap_final = softmax(y_cap)
	return np.argmax(y_cap_final, axis=1)		

##########################################################	
'''
Feature extraction
'''
#Loading the json data
with open('ModelParameters-F1-F2-F3-F4-Rating.json', 'r') as file_req:
	data_extraction = file_req.readlines()
	
data_extraction = map(lambda p: p.rstrip(), data_extraction)
data_json_str = "[" + ','.join(data_extraction) + "]"
df = pd.read_json(data_json_str)

#Normalizing and modelling the data as rows and columns
temp_X = []
X = []
y = []
for column in df:
	temp_X.append(df[column][0]['F1'])
	temp_X.append(df[column][0]['F2']/float(100))
	temp_X.append(df[column][0]['F3']/float(100))
	temp_X.append(df[column][0]['F4'])
	X.append(temp_X)
	temp_X = []
	y.append(df[column][0]['Rating'])    	
numRows = np.array(X).shape[0]
temp_y = []

#Creating five class labels for output in range of 0-100
for i in range(0,numRows):
	if y[i] <= 20:
		temp_y.append([1,0,0,0,0])
	elif y[i] > 20 and y[i] <= 40:
		temp_y.append([0,1,0,0,0])
	elif y[i] > 40 and y[i] <= 60:
		temp_y.append([0,0,1,0,0])
	elif y[i] > 60 and y[i] <= 80:
		temp_y.append([0,0,0,1,0])
	else:
		temp_y.append([0,0,0,0,1])
	
#train data size - 2/3 size of whole dataset	
train_size = 2 * len(X) / 3
train_X = np.array(X[0:train_size])
train_y = np.array(temp_y[0:train_size])
test_X = np.array(X[train_size+1:numRows])
test_y = np.array(temp_y[train_size+1:numRows])

##########################################################

#initializg the weight vectors with random values
w1 = np.random.random((train_X.shape[1],num_nodes))
w2 = np.random.random((num_nodes,len(train_y[0])))

#training and testing for given number of iterations
for j in itr:
	fw1,fw2 = neural_model(train_X, train_y, w1, w2, j) #training
	y_cap_f = neural_predict(test_X,fw1,fw2) #testing

	#Checking if predicted and actual class label are same if so, increment correct
	correct=0
	for i in range(0,test_X.shape[0]):
		if np.argmax(test_y[i]) == y_cap_f[i]:
			correct+=1

	#To display the prediction accuracy percentage
	print (100 * (correct / float(test_X.shape[0]))),j




