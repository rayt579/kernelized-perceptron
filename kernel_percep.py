"""
Implementation of kernelized perceptron on protein sequence data.  
"""

import numpy as np

def common_subsequence(x_m,x,p):
	'''
	Returns the count of unique common subsequences of length p in strings
	xm and x
	'''
	n = len(x_m)
	m = len(x)
	explored_sub = []
	common_count = 0
	
	#  Find number of common subsequences between xm and x
	for i in xrange(n-p+1):
		sub = x_m[i:(i+p)]
		if sub not in explored_sub:
			explored_sub.append(sub)
			for j in xrange(m-p+1):
				if x[j:j+p] == sub:
					common_count += 1
					break
	return common_count

def string_kernel(x_mist,y_mist,x,p):
	'''
	Returns the dot product of the weight vector and feature map of string data
	'''
	# Initialization
	n = len(y_mist)
	dp = 0 

	# Returns sum
	for i in xrange(n):
		dp += y_mist[i] * common_subsequence(x_mist[i],x,p)

	return dp

def perceptron(data, labels, n_passes,p):
	'''
	Returns the x and y vectors after running perceptron
	in the training set. 
	'''
	# Initialize vectors to store data and labels 
	n = labels.size
	x_mistakes = [] 
	y_mistakes = [] 

	# Runs perceptron pass on kernelized data 
	for j in xrange(n_passes):
		for i in xrange(n):
			if labels[i] * string_kernel(x_mistakes,y_mistakes,data[i],p) <= 0:
				x_mistakes.append(data[i])
				y_mistakes.append(labels[i])
	return x_mistakes, y_mistakes

def perceptron_clf(x_mist,y_mist,data,p):
	'''
	Takes stored x and y vectors from perceptron algorithm, classifies new data
	and returns predicted labels
	'''
	n = data.size
	pred = []
	
	for i in xrange(n):
		if string_kernel(x_mist,y_mist, data[i],p) > 0:
			pred.append(1)
		else:
			pred.append(-1)
	return np.array(pred)

def calc_error(true_labels, pred_labels):
	'''
	Returns percent error of predicted labels
	'''
	assert true_labels.size == pred_labels.size
	n = true_labels.size

	# Calculate the error
	mis_count = 0.0
	for i in xrange(n):
		if pred_labels[i] != true_labels[i]:
			mis_count += 1.0
	return (mis_count/n)

def main():
	'''
	Runs kernelized perceptron with string kernel. Print test and training 
	error
	'''
	train = np.genfromtxt('data/train_data.txt',dtype='string')
	test = np.genfromtxt('data/test_data.txt',dtype='string')

	train_data = train[:,0]
	train_labels = train[:, 1].astype(np.float)
	test_data = test[:,0]
	test_labels = test[:,1].astype(np.float)

	'''
	Single pass perceptron on p = 3
	'''
	# Fit perceptron clf
	x_p, y_p = perceptron(train_data,train_labels,1,3)
	
	# Run perceptron on train data
	train_predictions = perceptron_clf(x_p,y_p,train_data,3)
	print "Training error for kernelized perceptron for p = 3: ", calc_error(train_labels,train_predictions) 

	# Run perceptron on test data
	test_predictions = perceptron_clf(x_p,y_p,test_data,3)
	print "Test error for kernelized perceptron for p = 3: ", calc_error(test_labels,test_predictions) 
	
	
	'''
	Single pass perceptron on p = 4
	'''
	"""
	# Fit perceptron clf
	x_p, y_p = perceptron(train_data,train_labels,1,4)
	
	# Run perceptron on train data
	train_predictions = perceptron_clf(x_p,y_p,train_data,4)
	print "Training error for kernelized perceptron for p = 4: ", calc_error(train_labels,train_predictions) 

	# Run perceptron on test data
	test_predictions = perceptron_clf(x_p,y_p,test_data,4)
	print "Test error for kernelized perceptron for p = 4: ", calc_error(test_labels,test_predictions) 
	"""
	
if __name__ == '__main__':
	main()