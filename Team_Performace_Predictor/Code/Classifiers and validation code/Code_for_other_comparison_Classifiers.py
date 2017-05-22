# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 01:29:49 2017

@author: AdityaPMishra
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets, svm
import json
import scipy.stats as st
from pprint import pprint
from sklearn.cluster import KMeans
from scipy.interpolate import spline
from pylab import *
from scipy.stats import iqr
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
FinalFeaturearray = []
FinalResultArray = []
ItemArray = []

with open('ModelParameters-F1-F2-F3-F4-Rating.json') as data_file:    
    data = json.load(data_file)
for item in data:
    arr = []
    arr.append(data[item]['F1'])
    arr.append(data[item]['F2'])
    arr.append(data[item]['F3'])
    arr.append(data[item]['F4'])
    if int(data[item]['Rating'])<20:
        FinalResultArray.append(0)
    if int(data[item]['Rating'])>=20 and int(data[item]['Rating'])< 40:
        FinalResultArray.append(1)
    if int(data[item]['Rating'])>=40 and int(data[item]['Rating'])< 60:
        FinalResultArray.append(2)
    if int(data[item]['Rating'])>=60 and int(data[item]['Rating'])< 80:
        FinalResultArray.append(3)
    if int(data[item]['Rating'])>=80:
        FinalResultArray.append(4)
    arr = np.array(arr)
    FinalFeaturearray.append(arr)
FinalFeaturearray = np.array(FinalFeaturearray)
print(len(FinalFeaturearray))
FinalResultArray = np.array(FinalResultArray)
print(len(FinalResultArray))
x = int(2*(len(FinalFeaturearray)/3))
ranarray = np.vsplit(FinalFeaturearray,3)
ftrainarray =np.vstack((ranarray[0],ranarray[1]))
ftestarray = np.array(ranarray[2])
ranarray = np.split(FinalResultArray,3)
rtrainarray = np.append(ranarray[0],ranarray[1])
rtestarray = np.array(ranarray[2])

logistic = linear_model.LogisticRegression(C=1e5,max_iter=1000000)
logistic.fit(ftrainarray, rtrainarray)
print("Train Error LR ->" + str(logistic.score(ftrainarray, rtrainarray)))
print("Test Error LR ->" + str(logistic.score(ftestarray, rtestarray)))

logistic = linear_model.SGDClassifier()
logistic.fit(ftrainarray, rtrainarray)
print("Train Error SGD ->" + str(logistic.score(ftrainarray, rtrainarray)))
print("Test Error SGD ->" + str(logistic.score(ftestarray, rtestarray)))

svmt = svm.SVC()
svmt.fit(ftrainarray, rtrainarray)
print("Train Error SVM ->" + str(svmt.score(ftrainarray, rtrainarray)))
print("Test Error SVM ->" + str(svmt.score(ftestarray, rtestarray)))

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,activation ='tanh', hidden_layer_sizes=(5), random_state=1)
clf.fit(ftrainarray,rtrainarray)
print("Train Error NN ->" + str(clf.score(ftrainarray, rtrainarray)))
print("Test Error NN ->" + str(clf.score(ftestarray, rtestarray)))

clf = GaussianNB()
clf.fit(ftrainarray, rtrainarray)
print("Train Error GNB ->" + str(clf.score(ftrainarray, rtrainarray)))
print("Test Error GNB ->" + str(clf.score(ftestarray, rtestarray)))

clf = tree.DecisionTreeClassifier()
clf = clf.fit(ftrainarray, rtrainarray)
print("Train Error DT ->" + str(clf.score(ftrainarray, rtrainarray)))
print("Test Error DT ->" + str(clf.score(ftestarray, rtestarray)))

clf = RandomForestClassifier(n_estimators=10)
clf = clf.fit(ftrainarray, rtrainarray)
print("Train Error RFC ->" + str(clf.score(ftrainarray, rtrainarray)))
print("Test Error RFC ->" + str(clf.score(ftestarray, rtestarray)))

accuracyarr=[]
logistic = linear_model.LogisticRegression(C=1e5,max_iter=1000000)
scores = cross_val_score(logistic,FinalFeaturearray, FinalResultArray, cv=10)
print("Cross Val Score LR->  %0.2f (+/- %0.2f)" %(scores.mean(), scores.std()*2))
accuracyarr.append(scores.mean())
logistic = linear_model.SGDClassifier()
scores = cross_val_score(logistic,FinalFeaturearray, FinalResultArray, cv=10)
print("Cross Val Score SGD ->  %0.2f (+/- %0.2f)" %(scores.mean(), scores.std()*2))
accuracyarr.append(scores.mean())
svmt = svm.SVC()
scores = cross_val_score(svmt,FinalFeaturearray, FinalResultArray, cv=10)
print("Cross Val Score SVM ->  %0.2f (+/- %0.2f)" %(scores.mean(), scores.std()*2))
accuracyarr.append(scores.mean())

clf = tree.DecisionTreeClassifier()
scores = cross_val_score(clf,FinalFeaturearray, FinalResultArray, cv=10)
print("Cross Val Score DT ->  %0.2f (+/- %0.2f)" %(scores.mean(), scores.std()*2))
accuracyarr.append(scores.mean())
clf = RandomForestClassifier(n_estimators=10)
scores = cross_val_score(clf,FinalFeaturearray, FinalResultArray, cv=10)
print("Cross Val Score RFC ->  %0.2f (+/- %0.2f)" %(scores.mean(), scores.std()*2))
accuracyarr.append(scores.mean())
accuracyarr.append(0.627)
accuracyarr.append(0.64)
accuracyarr.append(0.51)
ind = np.arange(8)
print(ind)
print(accuracyarr)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, accuracyarr, width, color='b')

ax.set_ylabel('Accuracy')
ax.set_xlabel('Classifier')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('LR','SGD','SVM','DT','RFC','GNB','1LNN','MLNN-5'))
plt.show()

plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
aer =['0.0000001','0.0000002','0.0000003','0.0000004','0.0000005','0.0000006','0.0000007']
ber=['59.0822972173','62.7827116637','65.1213735938','64.8845470693','64.2332741267','61.9242155121','56.8620485494']
plt.plot(aer,ber)
plt.show()



#GNB - 62.72