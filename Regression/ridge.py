import pandas as pd
import numpy as np
from math import ceil, floor
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt


trainData = np.genfromtxt('MNIST_15_15.csv', delimiter=',', dtype=int, skip_header=1)
testData = np.genfromtxt('MNIST_LABEL.csv', delimiter=',', dtype=int, skip_header=1)

def main():
	kf = KFold(n_splits = 10)
	kf.get_n_splits(trainData)
	KFold(n_splits=10, random_state=None, shuffle=False)

	inf = float("inf")
	thresholds = [-inf, -0.1, 0, 0.1, inf]
	accuracy = []
	true, positive = [], []

	for train_index, test_index in kf.split(trainData):
	    training = trainData[train_index]
	    testing = trainData[test_index]
	    # Min-Max Normalization
	    train_data = training / 255.0
	    test_data = testing / 255.0
	    train_label = testData[train_index]
	    test_label = testData[test_index]
	    ones = np.ones(len(train_data))
	    
	    lam = 5.457317198860891
	#     print(lam)
	    Xt = np.transpose(train_data)
	    lmbda_inv = lam*np.identity(len(Xt))
	    theInverse = np.linalg.inv(np.dot(Xt, train_data)+ lmbda_inv)
	    w = np.dot(np.dot(theInverse, Xt), train_label)
	    predictions = (np.array(np.dot(test_data, w)))
	    len_a = sum(test_label == 5)
	#     print(len_a)
	    len_b = len(test_label) - len_a
	    for thresh in thresholds:
	        tpr = 0
	        fpr = 0
	        tnr = 0
	        fnr = 0
	        for i, name in enumerate(predictions):
	            if test_label[i] == 5:
	                if name >= thresh:
	                    tpr += 1
	                else:
	                    fnr += 1
	#                 print(ceil(name))
	            if test_label[i] == 6:
	                if name < thresh:
	                    tnr += 1
	                else:
	                    fpr += 1
	#                 print(ceil(name))
	# #         print("\nThresh :", thresh)
	        true.append(tpr / len_a)
	        positive.append(fpr / len_b)
	#         print(tpr, tnr)
	        accuracy.append((tpr + tnr)/ len(predictions))
#         print(len(predictions))
                
                
# print(sum(accuracy) / len(accuracy))
# print(accuracy)


	print("Average accuracy:", sum(accuracy) / len(accuracy))

	# print(max(true))

	# print(true)

	plt.scatter(true, positive)
	plt.xlabel("TPR")
	plt.ylabel("FPR")
	# plt.xlim(0, 2)
	# plt.ylim(0, 2)
	plt.show()



if __name__ == '__main__':
	main()