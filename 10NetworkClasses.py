import numpy as np 
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:

	# layer initialization
	def __init__(self, n_inputs, n_neurons):
		# initialize weights and biases
		self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))

	# forward pass
	def forward(self, inputs):
		# calculate output values from inputs, weights, and biases
		self.output = np.dot(inputs, self.weights) + self.biases

		# remember inputs for backpropagation
		self.inputs = inputs

	# backward pass
	def backward(self, dvalues):
		# gradients on parameters
		self.dweights = np.dot(self.inputs.T, dvalues)
		self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

		# gradient on values
		self.dinputs = np.dot(dvalues, self.weights.T)


# rectified linear activation
class Activation_ReLU:

	# forward pass
	def forward(self, inputs):
		self.output = np.maximum(0, inputs)

		# remember inputs for backpropagation
		self.inputs = inputs
		self.output = np.maximum(0, inputs)

	# backward pass
	def backward(self, dvalues):
		# make and modify a copy of the original list of values
		self.dinputs = dvalues.copy()

		# zero gradient for negative input values
		self.dinputs[self.inputs <= 0] = 0
# softmax activation
class Activation_Softmax:

	# forward pass
	def forward(self, inputs):

		# get unnormalized probabilities
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

		# normalized them for each sample
		probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

		self.output = probabilities

	# backward pass
	def backward(self, dvalues):
		
		# create uninitialized array
		self.dinputs = np.empty_like(dvalues)

		# enumerate outputs and gradients
		for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):

			# flatten output array (2d matrix becomes layer in 3d column?)
			single_output = single_output.rehape(-1, 1)

			# calculate jacobian matrix of output
			jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

# common loss class
class Loss:

	# calulates the data and regularization losses given model output and ground truth values
	def calculate(self, output, y):

		# calculate sample losses
		sample_losses = self.forward(output, y)

		# calculate mean loss
		data_loss = np.mean(sample_losses)

		return data_loss

class Loss_CategoricalCrossentropy(Loss):

	# forward pass
	def forward(self, y_pred, y_true):

		# number of samples in a batch
		samples = len(y_pred)

		# clip data from both sides to prevent division by zero without ruining the average
		# (everything now goes from almost zero to almost 1)
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

		# probabilities for categorically labeled target values
		if len(y_true.shape) == 1:
			correct_confidences = y_pred_clipped[range(samples), y_true]

		# mask values for one-hot encoded labels
		elif len(y_true.shape) ==2:
			correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

		# losses
		negative_log_likelihoods = -np.log(correct_confidences)
		return negative_log_likelihoods

	# backward pass
	def backward(self, dvalues, y_true):
		# number of samples
		samples = len(dvalues)

		# number of labels per sample (calculated from first sample)
		labels = len(dvalues[0])

		# if labels are sparse, turn them into a one-hot vector
		if len(y_true.shape) == 1:
			y_true = np.eye(labels)[y_true]

		# calculate gradient
		self.dinputs = -y_true / dvalues

		# normalize gradient
		self.dinputs = self.dinputs / samples

# Softmax classifier - combined softmax activation and cross-entropy loss for faster backpropagation
class Activation_Softmax_Loss_CategoricalCrossentropy():

	# creates activation and loss function objects
	def __init__(self):
		self.activation = Activation_Softmax()
		self.loss = Loss_CategoricalCrossentropy()

	# forward pass
	def forward(self, inputs, y_true):
		# output layer's ctivation function
		self.activation.forward(inputs)

		# set the output
		self.output = self.activation.output

		# calculate and return loss value
		return self.loss.calculate(self.output,  y_true)

	# backward pass
	def backward(self, dvalues, y_true):

		# number of samples
		samples = len(dvalues)

		# if labels are one-hot encoded, turn them into discrete values
		if len(y_true.shape) == 2:
			y_true = np.argmax(y_true, axis=1)

		# copy so we can safely modify
		self.dinputs = dvalues.copy()

		# calculate gradient
		self.dinputs[range(samples), y_true] -= 1

		# normalize gradient
		self.dinputs = self.dinputs / samples

class Optimizer_SGD:

	# initialize optimizer, default learning rate = 1
	def __init__(self, learning_rate=1.0):
		self.learning_rate = learning_rate

	# update parameters
	def update_params(self, layer):
		layer.weights += -self.learning_rate * layer.dweights
		layer.biases += -self.learning_rate * layer.dbiases

# create data
X, y = spiral_data(samples = 100, classes = 3)

# create dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 64)

# dense1 activation
activation1 = Activation_ReLU()

# create second dense layer with 64 input features and 3 output values (1 output for each class from dataset)
dense2 = Layer_Dense(64, 3)

# create combined loss and activation classifier
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()

# create optimizer
optimizer = Optimizer_SGD()

# donald's bullshit (manually set learning rate to get accuracy up to .7 before 3000 epochs!)
optimizer.learning_rate = 2.51

# train in a loop
for epoch in range(10001):

	# perform a forward pass of sample data
	dense1.forward(X)
	activation1.forward(dense1.output)
	dense2.forward(activation1.output)

	loss = loss_activation.forward(dense2.output, y)

	# calculate accuracy from output of activation 2 and targets (y)
	predictions = np.argmax(loss_activation.output, axis=1)

	if len(y.shape) == 2:
		y = np.argmax(y, axis=1)

	accuracy = np.mean(predictions==y)

	# more of donald's bullshit
	if accuracy > 0.7 and optimizer.learning_rate > 2 ** (1.0 - accuracy):
		optimizer.learning_rate -= 2 ** (1.0 - accuracy)
	
	if not epoch % 500:
		print("Epoch: ", epoch)
		print("Learning rate: ", optimizer.learning_rate)
		# display loss value
		print("Loss: ", loss)
		# display the accuracy
		print("Accuracy: ", accuracy, "\n")

	# perform a backward pass
	loss_activation.backward(loss_activation.output, y)
	dense2.backward(loss_activation.dinputs)
	activation1.backward(dense2.dinputs)
	dense1.backward(activation1.dinputs)

	# update the weights and biases
	optimizer.update_params(dense1)
	optimizer.update_params(dense2)
