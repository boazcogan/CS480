import numpy as np
import generate_labels
from random import shuffle

def loadData():
    print("Loading train and test data...")
    # Load the data and generate labels for 7 and 9
    # Since it is basic perceptron, The labels are binary, 1 or 0
    # in this case, 7s are 1, 9s are 0
    all_7_data = generate_labels.LoadAllLabelsOfNumber(7)
    # all_7_labels = np.ones((len(all_7_data)))
    all_9_data = generate_labels.LoadAllLabelsOfNumber(9)
    # all_9_labels = np.zeros((len(all_9_data)))


    train7 = all_7_data[:(4 * len(all_7_data)) // 5]
    test7 = all_7_data[(4 * len(all_7_data)) // 5:]
    labelsTrain7 = np.ones(len(train7))
    labelsTest7 = np.ones(len(test7))



    train9 = all_9_data[:(4 * len(all_9_data)) // 5]
    test9 = all_9_data[(4 * len(all_9_data)) // 5:]
    labelsTrain9 = np.zeros(len(train9))
    labelsTest9 = np.zeros(len(test9))


    TRAIN_DATA = np.array(train7 + train9)
    TRAIN_LABELS = np.concatenate((labelsTrain7,labelsTrain9))

    TEST_DATA = np.array(test7 + test9)
    TEST_LABELS = np.concatenate((labelsTest7,labelsTest9))

    #TRAIN_LABELS = list(np.array(
        #np.concatenate((all_7_labels[:(4 * len(all_7_data)) // 5], all_9_labels[:(4 * len(all_9_data)) // 5]))))

    #TEST_DATA = np.array(test7 + test9)
    #TEST_LABELS = np.array(
    #    np.concatenate((all_7_labels[(4 * len(all_7_data)) // 5:], all_9_labels[(4 * len(all_9_data)) // 5:])))
    print("finished loading train and test data")
    return TRAIN_DATA, TRAIN_LABELS, TEST_DATA, TEST_LABELS


class Perceptron(object):
    def __init__(self, learning_rate, input_shape,epochs):
        self.lr = learning_rate
        self.weights = np.random.uniform(-.1,.1,size=input_shape)
        self.bias = -1
        self.epochs = epochs
        self.optimal_success_ratio = 0
        self.optimal_weights = 0

    def single_epoch(self, data, labels):
        itervals = [i for i in range(len(data))]
        shuffle(itervals)
        for i in itervals:
            prediction = (np.dot(data[i], self.weights)-self.bias) >= 0
            if prediction != labels[i]:
                if prediction:
                    self.weights -= (self.lr*np.array(data[i]))
                    self.bias -= (self.lr*self.bias)
                else:
                    self.weights += (self.lr*np.array(data[i]))
                    self.bias += (self.lr*self.bias)

    def train(self, data, labels, testData, testLabels):
        for i in range(self.epochs):
            self.single_epoch(data,labels)
            predictions = np.array(perceptron.predict(testData))
            correct = predictions == testLabels
            accuracy = np.sum(correct)/len(testLabels)
            if accuracy > self.optimal_success_ratio:
                self.optimal_success_ratio = accuracy
                self.optimal_weights = self.weights

    def set_weights(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def predict(self, data):
        predictions = []
        for i in range(len(data)):
            prediction = (np.dot(data[i], self.weights)-self.bias) >= 0
            predictions.append(prediction)
        #predictions = [((np.dot(data[i], self.weights)-1) >= 0) for i in range(len(data))]

        return predictions


TRAIN_DATA, TRAIN_LABELS, TEST_DATA, TEST_LABELS = loadData()
perceptron = Perceptron(0.1,6,1000)
perceptron.train(TRAIN_DATA,TRAIN_LABELS, TEST_DATA, TEST_LABELS)
# the following weights are known to work
# perceptron.set_weights([-6.37737244, 4.04913265, 46.1064285, -96.24999999, -666.65857, 15.02999999], -1576.899999999)
predictions = np.array(perceptron.predict(TEST_DATA))
correct = predictions==TEST_LABELS
accuracy = np.sum(correct)/len(TEST_LABELS)
print(accuracy)
print(perceptron.optimal_success_ratio)
print(perceptron.optimal_weights)