import numpy as np
import generate_labels
import tqdm

class Perceptron(object):

    def __init__(self, no_of_inputs, epoch=100, learning_rate=0.01):
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:])
        summation += self.weights[0]
        try:
            if summation > 0.0:
              activation = 1
            else:
              activation = 0
            return activation
        except ValueError:
            retVal = []
            for elem in summation:
                if elem>0:
                    retVal.append(1)
                else:
                    retVal.append(0)
            return retVal


    def train(self, training_inputs, labels):
        for _ in tqdm.tqdm(range(self.epoch)):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)


print("Loading train data...")
# Load the data and generate labels for 1 and 5
# Since it is basic perceptron, The labels are binary, 1 or 0
# in this case, 1s are 1, 5s are 0
all_1_data = generate_labels.LoadAllLabelsOfNumber(1)
all_1_labels = np.ones((len(all_1_data)))
all_5_data = generate_labels.LoadAllLabelsOfNumber(5)
all_5_labels = np.zeros((len(all_5_data)))

print('Creating perceptron...')
# Create a perceptron with 7 input neurons, 1000 epochs, and a lr of 0.1
perceptron = Perceptron(len(all_1_data[1]),epoch=1000, learning_rate=0.1)

# Concatenate both training sets into one set
TRAIN_DATA = np.concatenate((all_1_data,all_5_data))
TRAIN_LABELS = np.concatenate((all_1_labels,all_5_labels))

print("Training...")
# Train the perceptron
perceptron.train(TRAIN_DATA, TRAIN_LABELS)
print("Training complete...")

# The perceptron has been trained, test it on 7 and 9.
# note we dont know which label will be assigned to 7 or 9 (although
# we can probably assume that 7 is 1 and 9 is 5) therefore we are
# checking to see if the labels are different from one another.

# load the data
print("Loading test data...")
all_7_data = generate_labels.LoadAllLabelsOfNumber(7)
all_9_data = generate_labels.LoadAllLabelsOfNumber(9)

# the project specifications say that only 20% of each dataset should be
# included in the training process.
all_7_data = all_7_data[:(len(all_7_data)//5)]
all_9_data = all_9_data[:(len(all_9_data)//5)]

print("Testing...")
# generate the labels
all_7_predicted = perceptron.predict(np.array(all_7_data))
all_9_predicted = perceptron.predict(np.array(all_9_data))
print("Testing complete")

# determine the associated label with the number
label7 = int((np.sum(all_7_predicted)/len(all_7_predicted))>0.5)
label9 = label7-1

if label7:
    label7 = np.ones(len(all_7_predicted))
    label9 = np.zeros(len(all_9_predicted))
else:
    label7 = np.zeros(len(all_7_predicted))
    label9 = np.ones(len(all_9_predicted))


# determine the success ratio
# (success of 7 + success of 9)/total test

success7 = np.sum(label7 == all_7_predicted)
success9 = np.sum(label9 == all_9_predicted)
total = len(all_9_predicted)+len(all_7_predicted)
ratio = (success7+success9)/total
print("The success ration is: ", ratio)
