#continue at
#https://pub.towardsai.net/building-neural-networks-from-scratch-with-python-code-and-math-in-detail-i-536fae5d7bbf#3a44
#you're in section 10.8 /10.9

import numpy as np

#define input
input_features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

#print(input_features.shape)
#input_features

#define target output
target_output = np.array([0, 1, 1, 1])
target_output = target_output.reshape(4, 1)

#each input has two features so we'll assign two random weights
weights = np.array([[0.1], [0.2]])

#assign bias and learning rate
bias = 0.3
lr = 0.05

#define a sigmoid function
def sigmoid(x):
	return 1/(1 + np.exp(-x))

#take the derivitabe for use in gradient descent
def sig_deriv(x):
	return sigmoid(x) * (1-sigmoid(x))

#our logic loop
for epoch in range(10000):
	inputs = input_features

	#feedfowrward input
	in_o = np.dot(inputs, weights) + bias

	#feedforward output
	out_o = sigmoid(in_o)

	#error calculations (i do not understand)
	error = out_o - target_output

	x = error.sum()
	#print(x)

	derror_douto = error
	douto_dino = sig_deriv(out_o)

	deriv = derror_douto * douto_dino

	#more math i need to wrap my head around
	inputs = input_features.T 
	deriv_final = np.dot(inputs, deriv)

	weights -= lr * deriv_final

	for i in deriv:
		bias -= lr * i

print(weights)
print(bias)

#section 10.9 predicting values
#taking inputs:
single_point = np.array([0,0])

#first step
result1 = np.dot(single_point, weights) + bias

#second step
result2 = sigmoid(result1)

print(f"final result = {result2}")