import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd

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

##########################################################
'''
Training
'''
#calculating total number of occurences of each class label
count = [0,0,0,0,0]
for i in range(0,train_size):
	if train_y[i][0] == 1:
		count[0]+=1
	if train_y[i][1] == 1:
		count[1]+=1
	if train_y[i][2] == 1:
		count[2]+=1
	if train_y[i][3] == 1:
		count[3]+=1
	if train_y[i][4] == 1:
		count[4]+=1		

#finding probability of each class label
for i in range(0,len(count)):
	count[i] = count[i] / float(train_size);

temp_p1=[]
temp_p2=[]
temp_p3=[]
temp_p4=[]
temp_p5=[]

#Creating list of input features belonging to different classes
for i in range(0,train_size):
	if train_y[i][0] == 1:
		temp_p1.append(train_X[i])	
	if train_y[i][1] == 1:
		temp_p2.append(train_X[i])
	if train_y[i][2] == 1:
		temp_p3.append(train_X[i])
	if train_y[i][3] == 1:
		temp_p4.append(train_X[i])
	if train_y[i][4] == 1:
		temp_p5.append(train_X[i])			

#converting to arrays
temp_p1 = np.array(temp_p1)
temp_p2 = np.array(temp_p2)
temp_p3 = np.array(temp_p3)
temp_p4 = np.array(temp_p4)
temp_p5 = np.array(temp_p5)

#calculating the mean of input feature given a class label
mean1 = temp_p1.mean(axis=0)
mean2 = temp_p2.mean(axis=0)
mean3 = temp_p3.mean(axis=0)
mean4 = temp_p4.mean(axis=0)
mean5 = temp_p5.mean(axis=0)

#calculating the variance of input feature given a class label
var1 = temp_p1.var(axis=0)
var2 = temp_p2.var(axis=0)
var3 = temp_p3.var(axis=0)
var4 = temp_p4.var(axis=0)
var5 = temp_p5.var(axis=0)
##########################################################

##########################################################
'''
Testing
'''
#calculating probability using Gaussian distribution given mean and variance
def func(x,mean,var):
	pi = 3.1415926
	denom = (2*pi*var)**.5
	num = math.exp(-(float(x)-float(mean))**2/(2*var))
	if np.abs(num/denom) == 0:
		return 1
	return np.abs(num/denom)

correct=0
for i in range(0,test_X.shape[0]):
	f1 = test_X[i][0]           #input feature 1
	f2 = test_X[i][1]			#input feature 2
	f3 = test_X[i][2]			#input feature 3
	f4 = test_X[i][3]			#input feature 4

	#calculating the gaussian probability of input feature vector belonging to first class
	p0 = math.log(func(f1,mean1[0],var1[0]),2) + math.log(func(f2,mean1[1],var1[1]),2) + math.log(func(f3,mean1[2],var1[2]),2) + math.log(func(f4,mean1[3],var1[3]),2) + math.log(count[0]+1,2)
	
	#calculating the gaussian probability of input feature vector belonging to second class
	p1 = math.log(func(f1,mean2[0],var2[0]),2) + math.log(func(f2,mean2[1],var2[1]),2) + math.log(func(f3,mean2[2],var2[2]),2) + math.log(func(f4,mean2[3],var2[3]),2) + math.log(count[1]+1,2)

	#calculating the gaussian probability of input feature vector belonging to third class
	p2 = math.log(func(f1,mean3[0],var3[0]),2) + math.log(func(f2,mean3[1],var3[1]),2) + math.log(func(f3,mean3[2],var3[2]),2) + math.log(func(f4,mean3[3],var3[3]),2) + math.log(count[2]+1,2)

	#calculating the gaussian probability of input feature vector belonging to fourth class
	p3 = math.log(func(f1,mean4[0],var4[0]),2) + math.log(func(f2,mean4[1],var4[1]),2) + math.log(func(f3,mean4[2],var4[2]),2) + math.log(func(f4,mean4[3],var4[3]),2) + math.log(count[3]+1,2)
	
	#calculating the gaussian probability of input feature vector belonging to fifth class	
	p4 = math.log(func(f1,mean5[0],var5[0]),2) + math.log(func(f2,mean5[1],var5[1]),2) + math.log(func(f3,mean5[2],var5[2]),2) + math.log(func(f4,mean5[3],var5[3]),2) + math.log(count[4]+1,2)
	
	#creating the list of probabilities calculated
	l = []
	l.append(p0);
	l.append(p1);
	l.append(p2);
	l.append(p3);
	l.append(p4);
	
	#finding the maximum probability i.e., predicted class and if is equal to actual class => correct prediction
	if test_y[i][np.argmax(l)]==1:
		correct+=1

##########################################################

#To display the prediction accuracy percentage
print (100 * (correct / float(test_X.shape[0])))
		













