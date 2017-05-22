import sys
import numpy as np
import json
import random
from ANN_singleLayer import singleLayer


def trainData(train_data):
	return singleLayer(train_data)


def convertToArray(json_data):
	outer_arr = []
	for json in json_data:
		inner_arr = []
		#print json_data[json]
		inner_arr.append(json)
		for entry in json_data[json]:
			#inner_arr.append(json_data[json])
			inner_arr.append(json_data[json][entry])
		#print json_data[json]
		outer_arr.append(inner_arr)
	#print outer_arr
	return outer_arr

def readData(filePath):
	data = {}
	arr_data = []
	with open(filePath) as json_data:
		data = json.load(json_data)
	arr_data = convertToArray(data)
	return arr_data

def normalize(f1,f2,f3,f4):
    return f1/100, f2/100, f3/100, f4/100


def generateRandomSegmentIndex(len_data, k):
	len_seg = len_data/k
	start_index = random.randrange(0, (len_data - len_seg + 1 ))
	end_index = ( start_index + len_seg - 1 )
	return start_index, end_index

def generateArrays(train_data, k):
	start_index, end_index = generateRandomSegmentIndex(len(train_data), k)

	#print train_data
	test_ar = train_data[start_index : end_index +1]
	new_train_ar_one = train_data[:start_index] 
	new_train_ar_two = train_data[end_index+1:]

	new_train_ar = []
	new_train_ar.extend(new_train_ar_one)
	new_train_ar.extend(new_train_ar_two)
	return new_train_ar, test_ar

def testData(test_ar, weights_ar):

	w0 = weights_ar[0]
	w1 = weights_ar[1]
	w2 = weights_ar[2]
	w3 = weights_ar[3]
	w4 = weights_ar[4]

	pred_labels = []
	
	for i in range(0, len(test_ar)):
		f1 = float(test_ar[i][1])
		f2 = float(test_ar[i][2])
		f3 = float(test_ar[i][3])
		f4 = float(test_ar[i][4])

		pred_label = w0 + w1*f1 + w2*f2 + w3*f3 + w4*f4
		pred_labels.append(pred_label)

	return pred_labels


def calculateAccuracy(predictedLabels, test_ar):

	print 'predicted labels length', len(predictedLabels)
	print 'test_ar length', len(test_ar)
	if len(predictedLabels)!=len(test_ar):
		print 'Length Mismatch!! Something wrong!'
		sys.exit('Length Mismatch!! Something wrong!')

	correctCount = 0
	for i in range(0, len(test_ar)):
		true_label = test_ar[i][5]
		print 'true label->', true_label
		print 'predicted label-->', predictedLabels[i]
		if true_label == predictedLabels[i]:
			correctCount+=1

	total_accuracy = correctCount/(len(test_ar))
	return total_accuracy 

def kCrossValidate(train_data, k, num_of_iter):

	accuracy = []
	#iterate num_of_iter times
	for i in range(0, num_of_iter+1):

		#initialize weights array for current iteration
		print 'iteration number -->', i
		weights_ar = []

		#choose the (k-1)th part for current iteration
		print 'generating arrays...'
		new_train_ar, test_ar = generateArrays(train_data, k)

		print 'new train data array length-->', len(new_train_ar)
		print 'new test data array length-->', len(test_ar)

		#train data to generate weights
		print 'training data...'
		#weights_ar = trainData(new_train_ar)

		weights_ar = [0.86583311399, -0.348257506191, -0.0158227870319, 0.0645269636937, 0.75572762318]
		#test data
		print 'testing data'
		predictedLabels = testData(test_ar, weights_ar)

		#calculate accuracy
		print 'predicting accuracy' 
		predictedAccuracy = calculateAccuracy(predictedLabels, test_ar)

		#save the result from the iteration
		accuracy.append(predictedAccuracy)

	#return average accuracy
	avg_accuracy = np.average(accuracy)
	return avg_accuracy

def main():
	#initialize data
	num_of_iter = 10    #number of iterations
	k = 10				#value for k , cross validation parameter
	
	#read data from the file and get it as an array
	#data = readData('ratings.json')
	data = readData('result_new.json')
	#data = simplejson.loads('result_new.json')
	#print data
	avg_accuracy = kCrossValidate(data, k, num_of_iter)
	print 'average accuracy-->', avg_accuracy
    #print len(split_data), len(split_data[0]), len(split_data[1])

if __name__ == "__main__":
    main()
			
