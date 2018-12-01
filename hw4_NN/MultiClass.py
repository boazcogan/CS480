import numpy as np
class MulticlassPerceptron:
	def __init__(self, weights, eta):
		self.weights = weights
		self.eta = eta

	def ActivationLabel(self, input):
		results = np.dot(self.weights, input)
		return np.argmax(results)


	def UpdateWeights(self, input, predicted_label, target_label):
		for i in range(len(self.weights[target_label])):
			self.weights[target_label][i] = self.weights[target_label][i] + self.eta * input[i]

		for i in range(len(self.weights[predicted_label])):
			self.weights[predicted_label][i] = self.weights[predicted_label][i] - self.eta * input[i]

def DoTraining(train_inputs, train_targets):
	weights = np.random.uniform(-0.1, 0.1, [10, len(train_inputs[0])])
	eta = 0.1
	n_epochs = 1000

	p = MulticlassPerceptron(weights, eta)

	for stop in range(n_epochs):# and not AllWeightsValid(p, train_inputs, train_targets)):
		if stop % 100 == 0:
			print("Gone through", stop, "epochs on the TRAIN set")

		for i in range(len(train_inputs)):
			activation = p.ActivationLabel(train_inputs[i])

			if activation != train_targets[i]:
				p.UpdateWeights(train_inputs[i], activation, train_targets[i])

	return p