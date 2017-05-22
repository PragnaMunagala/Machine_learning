import sys
import numpy as np
import json
import math

def convertToArray(json_data):
	outer_arr = []
	for json in json_data:
		inner_arr = []
		inner_arr.append(json)
		for entry in json_data[json]:
			inner_arr.append(json_data[json][entry])
		outer_arr.append(inner_arr)
	return outer_arr

def readData(filePath):
	with open(filePath) as json_data:
		data = json.load(json_data)
	arr_data = convertToArray(data)
	return arr_data

def getClassLabel(label):
    classLabel = -1
    if label <= 20 and label >= 0:
        classLabel = 1
    elif label <= 40 and label > 20:
        classLabel = 2
    elif label <= 60 and label > 40:
        classLabel = 3
    elif label <= 80 and label > 60:
        classLabel = 4
    elif label <= 100 and label > 80:
        classLabel = 5
    return classLabel

def singleLayer(training_data):
    learning_rate = 0.0000005

    # Errors in percentage
    avg_error = 1000000000
    error_threshold = 0.00000005

    # Initialize weights
    w0 = 0.30
    w1 = 0.45
    w2 = 0.26
    w3 = 0.62
    w4 = 0.34
    iteration = 0
    avg_error_copy = -1
    while (iteration != 4800 and avg_error > error_threshold) and \
            (w0==w0 or w1 == w1 or w2 == w2 or w3 == w3 or w4 == w4):
        w0_gradient = w1_gradient = w2_gradient = w3_gradient = w4_gradient = 0.0
        iteration += 1

        total_error = 0.0
        for i in range(2*len(training_data)/3):
            size = (2*len(training_data)/3)
            # Get features
            f1= training_data[i][1]
            f2 = training_data[i][2]
            f3 = training_data[i][3]
            f4 = training_data[i][4]
            true_label = getClassLabel(training_data[i][5])

            # Calculate predicted label for the given feature set
            pred_label = getClassLabel(w0 + w1*f1 + w2*f2 + w3*f3 + w4*f4)

            # Calculate mean squared error
            total_error += 0.5 * (((pred_label - true_label))**2) / size

            # Calculate gradient for weights normalizations done to maintain the values of weights below one
            w0_gradient += (pred_label - true_label) / float(size)
            w1_gradient += (pred_label - true_label) * f1 / float(size)
            w2_gradient += (pred_label - true_label) * f2 / float(size)
            w3_gradient += (pred_label - true_label) * f3 / float(size)
            w4_gradient += (pred_label - true_label) * f4 / float(size)

        w0 = w0 - learning_rate * w0_gradient
        w1 = w1 - learning_rate * w1_gradient
        w2 = w2 - learning_rate * w2_gradient
        w3 = w3 - learning_rate * w3_gradient
        w4 = w4 - learning_rate * w4_gradient
        avg_error_copy = avg_error
        avg_error = total_error

    print "final_weights : ", w0, w1, w2, w3, w4
    print "iterations : ", iteration

    correctPred = 0
    for i in range(int(2*len(training_data)/3) , len(training_data)):
        # Get features
        f1 = training_data[i][1]
        f2 = training_data[i][2]
        f3 = training_data[i][3]
        f4 = training_data[i][4]
        true_label = getClassLabel(training_data[i][5])
        pred_label = getClassLabel(w0 + w1*f1 + w2*f2 + w3*f3 + w4*f4)
        if true_label == pred_label:
            correctPred += 1
    accuracy = float(correctPred) * 100 / (len(training_data) - int(2*len(training_data)/3))
    print "Validation Accuracy:", accuracy


def main():
    fileNameAndPath = "C:\\Users\\Shalmali\\PycharmProjects\\SML\\ModelParameters-F1-F2-F3-F4-Rating.json"
    data = readData(fileNameAndPath)
    singleLayer(data)

if __name__ == "__main__":
    main()