# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np
import generate_labels
from random import shuffle
import tqdm


class mlp:
    """ A Multi-Layer Perceptron"""

    def __init__(self, inputs, targets, nhidden, beta=1, momentum=0.9, outtype='logistic'):
        """ Constructor """
        # Set up network size
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhidden = nhidden

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype

        # Initialise network
        self.weights1 = (np.random.rand(self.nin + 1, self.nhidden) - 0.5) * 2 / np.sqrt(self.nin)
        self.weights2 = (np.random.rand(self.nhidden + 1, self.nout) - 0.5) * 2 / np.sqrt(self.nhidden)

        self.optimalWeights1 = self.weights1
        self.optimalWeights2 = self.weights2
        self.optimalEpoch = 0
        self.optimal = 0

    def earlystopping(self, inputs, targets, valid, validtargets, eta, niterations=100):

        valid = np.concatenate((valid, -np.ones((np.shape(valid)[0], 1))), axis=1)

        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000

        count = 0
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1) > 0.001)):
            count += 1
            print(count)
            self.mlptrain(inputs, targets, eta, niterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.mlpfwd(valid)
            new_val_error = 0.5 * np.sum((validtargets - validout) ** 2)

        print("Stopped", new_val_error, old_val_error1, old_val_error2)
        return new_val_error

    def mlptrain(self, inputs, targets, eta, niterations, test_inputs, TEST_LABELS):
        """ Train the thing """
        # Add the inputs that match the bias node

        inputs = np.concatenate((inputs, -np.ones((self.ndata, 1))), axis=1)
        change = range(self.ndata)

        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))

        for n in range(niterations):

            self.outputs = self.mlpfwd(inputs)

            error = 0.5 * np.sum((self.outputs - targets) ** 2)
            if (np.mod(n, 100) == 0):
                print("Iteration: ", n, " Error: ", error)

            # Different types of output neurons
            if self.outtype == 'linear':
                deltao = (self.outputs - targets) / self.ndata
            elif self.outtype == 'logistic':
                deltao = self.beta * (targets-self.outputs)*self.outputs*(1-self.outputs)
            #    deltao = self.beta * (self.outputs - targets) * self.outputs * (1.0 - self.outputs)
            #
            #elif self.outtype == 'logistic':
            #    deltao = self.beta * (self.outputs - targets) * self.outputs * (1.0 - self.outputs)
            elif self.outtype == 'softmax':
                deltao = (self.outputs - targets) * (self.outputs * (-self.outputs) + self.outputs) / self.ndata
            else:
                print("error")

            deltah = self.hidden * self.beta * (1.0 - self.hidden) * (np.dot(deltao, np.transpose(self.weights2)))

            updatew1 = eta * (np.dot(np.transpose(inputs), deltah[:, :-1])) + self.momentum * updatew1
            updatew2 = eta * (np.dot(np.transpose(self.hidden), deltao)) + self.momentum * updatew2
            self.weights1 -= updatew1
            self.weights2 -= updatew2
            self.mlpfwd(test_inputs, mode="test", ALL_TEST_LABELS=TEST_LABELS, current_epoch=i)

            # Randomise order of inputs (not necessary for matrix-based calculation)
            # np.random.shuffle(change)
            # inputs = inputs[change,:]
            # targets = targets[change,:]

    def mlpGetOptimal(self):
        return self.optimal, self.optimalEpoch, self.optimalWeights1, self.optimalWeights2

    def mlpfwd(self, inputs, mode="train", ALL_TEST_LABELS=[], current_epoch=0):
        """ Run the network forward """
        if mode=="train":
            self.hidden = np.dot(inputs, self.weights1);
            self.hidden = 1.0 / (1.0 + np.exp(-self.beta * self.hidden))
            self.hidden = np.concatenate((self.hidden, -np.ones((np.shape(inputs)[0], 1))), axis=1)

            outputs = np.dot(self.hidden, self.weights2);

            # Different types of output neurons
            if self.outtype == 'linear':
                return outputs
            elif self.outtype == 'logistic':
                return 1.0 / (1.0 + np.exp(-self.beta * outputs))
            elif self.outtype == 'softmax':
                normalisers = np.sum(np.exp(outputs), axis=1) * np.ones((1, np.shape(outputs)[0]))
                return np.transpose(np.transpose(np.exp(outputs)) / normalisers)
            else:
                print("error")
        elif mode=="test":
            hidden = np.dot(inputs, self.weights1);
            hidden = 1.0 / (1.0 + np.exp(-self.beta * hidden))
            hidden = np.concatenate((hidden, -np.ones((np.shape(inputs)[0], 1))), axis=1)

            outputs = np.dot(hidden, self.weights2);

            # Different types of output neurons
            if self.outtype == 'linear':
                accuracy = outputs==np.argmax(ALL_TEST_LABELS,axis=1)
                accuracy = sum(accuracy)/len(ALL_TEST_LABELS)
                if accuracy>self.optimal:
                    self.optimal = accuracy
                    self.optimalWeights1 = self.weights1
                    self.optimalWeights2 = self.weights2
                    self.optimalEpoch = current_epoch
                return outputs
            elif self.outtype == 'logistic':
                outputs = 1.0 / (1.0 + np.exp(-self.beta * outputs))
                accuracy = outputs==np.argmax(ALL_TEST_LABELS,axis=1)
                accuracy = sum(accuracy)/len(ALL_TEST_LABELS)
                if accuracy>self.optimal:
                    self.optimal = accuracy
                    self.optimalWeights1 = self.weights1
                    self.optimalWeights2 = self.weights2
                    self.optimalEpoch = current_epoch
                return outputs
            elif self.outtype == 'softmax':
                normalisers = np.sum(np.exp(outputs), axis=1) * np.ones((1, np.shape(outputs)[0]))
                outputs = np.transpose(np.transpose(np.exp(outputs)) / normalisers)
                accuracy = np.argmax(outputs,axis=1)==np.argmax(ALL_TEST_LABELS,axis=1)
                accuracy = sum(accuracy)/len(ALL_TEST_LABELS)
                if accuracy>self.optimal:
                    self.optimal = accuracy
                    self.optimalWeights1 = self.weights1
                    self.optimalWeights2 = self.weights2
                    self.optimalEpoch = current_epoch
                return outputs
            else:
                print("error")
        else:
            print("error")


    def confmat(self, inputs, targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs, -np.ones((np.shape(inputs)[0], 1))), axis=1)
        outputs = self.mlpfwd(inputs)

        nclasses = np.shape(targets)[1]

        if nclasses == 1:
            nclasses = 2
            outputs = np.where(outputs > 0.5, 1, 0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        cm = np.zeros((nclasses, nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0) * np.where(targets == j, 1, 0))

        print("Confusion matrix is:")
        print(cm)
        print("Percentage Correct: ", np.trace(cm) / np.sum(cm) * 100)
        return cm

    def setOptimal(self, optimalWeights1, optimalWeights2):
        self.weights1 = optimalWeights1
        self.weights2 = optimalWeights2


def LoadData():
    # start by loading all of the labels and their datasets
    ALL_TEST = []
    ALL_TEST_LABELS = []
    ALL_TRAIN_LABELS = []
    ALL_TRAIN = []
    # TODO:
    for i in tqdm.tqdm(range(10)):
        currentFeatures = generate_labels.Part3LoadAllFeaturesOfNumber(i)#LoadAllLabelsOfNumber(i)
        fourFiths = round(4/5*len(currentFeatures))
        ALL_TRAIN+=currentFeatures[:fourFiths]
        ALL_TEST+=currentFeatures[fourFiths:]
        currentLabel = [i for j in range(len(currentFeatures))]
        ALL_TRAIN_LABELS+=currentLabel[:fourFiths]
        ALL_TEST_LABELS+=currentLabel[fourFiths:]

    ALL_TEST, ALL_TEST_LABELS = scramble(ALL_TEST, ALL_TEST_LABELS)
    ALL_TRAIN, ALL_TRAIN_LABELS = scramble(ALL_TRAIN, ALL_TRAIN_LABELS)

    return ALL_TRAIN, ALL_TEST, ALL_TRAIN_LABELS, ALL_TEST_LABELS

def scramble(Vals, Labels):
    key = [i for i in range(len(Vals))]
    shuffle(key)
    tempVals = []
    tempLabels = []
    for elem in key:
        tempVals.append(Vals[elem])
        tempLabels.append(Labels[elem])
    return tempVals, tempLabels

ALL_TRAIN1, ALL_TEST1, ALL_TRAIN_LABELS, ALL_TEST_LABELS = LoadData()
ALL_TRAIN, ALL_TEST, ALL_TRAIN_LABELS, ALL_TEST_LABELS = np.array(ALL_TRAIN1),np.array(ALL_TEST1),np.array(ALL_TRAIN_LABELS),np.array(ALL_TEST_LABELS)


NEW_TRAIN_LABELS = []
NEW_TEST_LABELS= []
for i in range(len(ALL_TRAIN_LABELS)):
    NEW_TRAIN_LABELS.append(np.zeros(10))
    NEW_TRAIN_LABELS[i][ALL_TRAIN_LABELS[i]] = 1

for i in range(len(ALL_TEST_LABELS)):
    NEW_TEST_LABELS.append(np.zeros(10))
    NEW_TEST_LABELS[i][ALL_TEST_LABELS[i]] = 1

ALL_TRAIN_LABELS = np.array(NEW_TRAIN_LABELS)
ALL_TEST_LABELS = np.array(NEW_TEST_LABELS)


anddata = np.array([[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
xordata = np.array([[0,0,0],[0,1,1],[1,0,1],[1,1,0]])

labelFormatexpected = anddata[:,2:3]
inputFormatExpected = anddata[:,0:2]
#ALL_TRAIN_LABELS = ALL_TRAIN_LABELS.reshape(ALL_TRAIN_LABELS.shape[0],-1)
#ALL_TEST_LABELS = ALL_TEST_LABELS.reshape(ALL_TEST_LABELS.shape[0],-1)


optimal_hidden = 0
optimal_accuracy = 0
for i in [10]:
    ALL_TRAIN, ALL_TEST = np.array(ALL_TRAIN1),np.array(ALL_TEST1)
    perceptron = mlp(ALL_TRAIN,ALL_TRAIN_LABELS,i,1,1, "softmax")
    TEST_SHAPE = np.shape(ALL_TEST)[0]
    ALL_TEST = np.concatenate((ALL_TEST, -np.ones((TEST_SHAPE, 1))), axis=1)
    perceptron.mlptrain(ALL_TRAIN,ALL_TRAIN_LABELS,0.1,1000, ALL_TEST, ALL_TEST_LABELS)
    values1,values2,values3,values4 = perceptron.mlpGetOptimal()
    if values1>optimal_accuracy:
        optimal_accuracy=values1
        optimal_hidden=i
        optimal_epoch = values2
        optimal_weights_1 = values3
        optimal_weights_2 = values4

ALL_TRAIN, ALL_TEST = np.array(ALL_TRAIN1),np.array(ALL_TEST1)
#TEST_SHAPE = np.shape(ALL_TEST)[0]
#ALL_TEST = np.concatenate((ALL_TEST, -np.ones((TEST_SHAPE, 1))), axis=1)
perceptron = mlp(ALL_TRAIN,ALL_TRAIN_LABELS,optimal_hidden,1,1, "softmax")
perceptron.setOptimal(optimal_weights_1,optimal_weights_2)
cm = perceptron.confmat(ALL_TEST, ALL_TEST_LABELS)
outfile = open("Part3Output.txt", "w")
outfile.write("Number of Neurons in HiddenLayer that gave minimum error: " + str(optimal_hidden))
outfile.write("\nEpoch at which the minimum error was reached: "+str(optimal_epoch))
outfile.write("\nThe error rate as a fraction between 0 and 1: "+str(optimal_accuracy))
outfile.write("\nWeights of all the hidden-layer neurons as well as the output neurons of the optimal network:\n")
outfile.write("HiddenLayer: \n", +str(optimal_weights_1))
outfile.write("\nOutputLayer: \n"+str(optimal_weights_2))
outfile.write("\n\nConfusionMatrix: \n"+ str(cm))
outfile.close()