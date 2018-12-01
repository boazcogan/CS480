import numpy as np
import tqdm
from random import shuffle
import generate_labels
from MultiClass import *


def scramble(Vals, Labels):
	key = [i for i in range(len(Vals))]
	shuffle(key)
	tempVals = []
	tempLabels = []
	for elem in key:
		tempVals.append(Vals[elem])
		tempLabels.append(Labels[elem])
	return tempVals, tempLabels


def LoadData():
	# start by loading all of the labels and their datasets
	ALL_TEST = []
	ALL_TEST_LABELS = []
	ALL_TRAIN_LABELS = []
	ALL_TRAIN = []
	for i in tqdm.tqdm(range(10)):
		currentFeatures = generate_labels.Part3LoadAllFeaturesOfNumber(i)  # LoadAllLabelsOfNumber(i)
		fourFiths = round(4 / 5 * len(currentFeatures))
		ALL_TRAIN += currentFeatures[:fourFiths]
		ALL_TEST += currentFeatures[fourFiths:]
		currentLabel = [i for j in range(len(currentFeatures))]
		ALL_TRAIN_LABELS += currentLabel[:fourFiths]
		ALL_TEST_LABELS += currentLabel[fourFiths:]

	ALL_TEST, ALL_TEST_LABELS = scramble(ALL_TEST, ALL_TEST_LABELS)
	ALL_TRAIN, ALL_TRAIN_LABELS = scramble(ALL_TRAIN, ALL_TRAIN_LABELS)

	return ALL_TRAIN, ALL_TEST, ALL_TRAIN_LABELS, ALL_TEST_LABELS


def sigmoid(x):
	x = np.array(x, dtype=np.float128)
	return 1 / (1 + np.exp(-x))


def sigmoidDerivative(x):
	x = np.array(x, dtype=np.float128)
	return x * (1 - x)


class MultiLayerPerceptron(object):
	def __init__(self, InputShape, numHidden, outputShape, Bias=-1, LR=0.1):
		self.hiddenNeurons = np.random.uniform(-0.1, 0.1, (InputShape + 1, numHidden))
		self.outputNeurons = np.random.uniform(-0.1, 0.1, (numHidden + 1, outputShape))
		self.lr = LR
		self.bias = Bias
		self.optimal = 0
		self.optimalHidden = self.hiddenNeurons
		self.optimalOutput = self.outputNeurons
		self.numHidden = numHidden

	def forward_propogation(self, oneInputVector):
		# compute the activation of the input neurons
		inputValues = np.dot(np.insert(oneInputVector, 0, self.bias), self.hiddenNeurons)
		hiddenActivation = sigmoid(inputValues)

		# work through the network until you get to the output layer neurons
		outputValues = np.dot(np.insert(hiddenActivation, 0, self.bias), self.outputNeurons)
		outputActivation = sigmoid(outputValues)
		# print(inputActivation)
		# print(hiddenActivation)

		return hiddenActivation, outputActivation

	def backwards_propogation(self, outputActivation, hiddenActivation, Label, inputs):
		# compute error at outputLayer
		output_error = (Label - outputActivation) * outputActivation * (1 - outputActivation)

		hidden_error = sigmoidDerivative(hiddenActivation) * np.sum(output_error * self.outputNeurons)

		self.outputNeurons -= self.lr * output_error * hiddenActivation

		hiddenActivation = hiddenActivation.reshape(self.numHidden, 1)
		partOne = (hiddenActivation * np.insert(inputs, 0, self.bias))
		partTwo = (self.lr * partOne)
		self.hiddenNeurons -= partTwo.T

	def epoch(self, inputs, labels):
		for i in range(len(inputs)):
			hiddenActivation, outputActivation = self.forward_propogation(inputs[i])
			self.backwards_propogation(outputActivation, hiddenActivation, labels[i], inputs[i])

	def train(self, inputs, labels, epochs, TESTDATA, TESTLABELS):
		for _ in tqdm.tqdm(range(epochs)):
			self.epoch(inputs, labels)

			# evaluate
			correctlyClassified = 0
			for i in range(len(TESTDATA)):
				_, outputActivation = self.forward_propogation(TESTDATA[i])
				correctlyClassified += np.argmax(outputActivation) == TESTLABELS[i]
			if (correctlyClassified / len(TESTDATA)) > self.optimal:
				self.optimal = correctlyClassified / len(TESTDATA)
				self.optimalHidden = self.hiddenNeurons[:]
				self.optimalOutput = self.outputNeurons[:]
				print("learning is occuring: ", self.optimal)


ALL_TRAIN, ALL_TEST, ALL_TRAIN_LABELS, ALL_TEST_LABELS = LoadData()

p = DoTraining(ALL_TRAIN[:int(len(ALL_TRAIN))], ALL_TRAIN_LABELS[:int(len(ALL_TRAIN_LABELS))])
print(p.weights)

right = [0 for _ in range(10)]
wrong = right[:]

confusion_matrix = np.array([[0 for _ in range(10)] for _ in range(10)])

for i in range(int(len(ALL_TEST))):
	activation = p.ActivationLabel(ALL_TEST[i])
	confusion_matrix[activation][ALL_TEST_LABELS[i]] += 1

print(confusion_matrix)

# perceptron = MultiLayerPerceptron(49,15,10)
# perceptron.train(ALL_TRAIN, ALL_TRAIN_LABELS, 1000, ALL_TEST, ALL_TEST_LABELS)
# print(perceptron.optimal)
