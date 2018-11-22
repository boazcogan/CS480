'''
source: https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/
'''
import InputAndLabels
import numpy as np
Verbose = True


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidDerivative(x):
    return x*(1-x)

class NN(object):
    def __init__(self, epoch, lr, input_shape, hidden_layer_shape, output_shape):
        self.epoch = epoch
        self.Learning_Rate = lr
        self.hidden_layer_neurons = hidden_layer_shape
        self.output_neurons = output_shape
        self.input_layer_neurons = input_shape
        self.weight_hidden = np.random.uniform(size=(self.input_layer_neurons,self.hidden_layer_neurons))
        self.bias_hidden=np.random.uniform(size=(1,self.hidden_layer_neurons))
        self.weight_out=np.random.uniform(size=(self.hidden_layer_neurons,self.output_neurons))
        self.bias_output = np.random.uniform(size=(1,self.output_neurons))
        self.optimal_success_ratio = (0,0)

    def train_once(self, X, Y):
        hidden_layer_input1=np.dot(X,self.weight_hidden)
        hidden_layer_input=hidden_layer_input1 + self.bias_hidden
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input1=np.dot(hiddenlayer_activations,self.weight_out)
        output_layer_input= output_layer_input1 + self.bias_output
        output = sigmoid(output_layer_input)

        E = Y-output
        slope_output_layer = sigmoidDerivative(output)
        slope_hidden_layer = sigmoidDerivative(hiddenlayer_activations)
        d_output = E * slope_output_layer
        Error_at_hidden_layer = d_output.dot(self.weight_out.T)
        d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
        self.weight_out += hiddenlayer_activations.T.dot(d_output) * self.Learning_Rate
        self.bias_output += np.sum(d_output, axis=0,keepdims=True) * self.Learning_Rate
        self.weight_hidden += X.T.dot(d_hiddenlayer) * self.Learning_Rate
        self.bias_hidden += np.sum(d_hiddenlayer, axis=0,keepdims=True) * self.Learning_Rate

        return output

    def predict(self, X):
        hidden_layer_input1=np.dot(X,self.weight_hidden)
        hidden_layer_input=hidden_layer_input1 + self.bias_hidden
        hiddenlayer_activations = sigmoid(hidden_layer_input)
        output_layer_input1=np.dot(hiddenlayer_activations,self.weight_out)
        output_layer_input= output_layer_input1 + self.bias_output
        output = sigmoid(output_layer_input)
        return output

    def train(self, X, Y):
        for i in range(self.epoch):
            out = self.train_once(X,Y)
            self.update_success_ratio(out, Y, i)
        return self.optimal_success_ratio

    def update_success_ratio(self, output, labels, epoch):
        current_ratio = sum((output>0.5)==labels)/64
        if current_ratio > self.optimal_success_ratio[0]:
            self.optimal_success_ratio = (current_ratio, epoch)
            if Verbose:
                print("optimal success ratio of", self.optimal_success_ratio[0], "found at epoch:", self.optimal_success_ratio[1])
        return

X, Y = InputAndLabels.getValuesAndLabels()
X = np.array(X)
#Xtest = X[50:]
#X = X[:50]

Y = [[Y[i]] for i in range(len(Y))]
Y = np.array(Y)
#Ytest = Y[50:]
#Y = Y[:50]


learning_rate = 0.01
NeuralNet = NN(1000, learning_rate, X.shape[1], 2, 1)
success_ratio = NeuralNet.train(X,Y)
final_prediction = (NeuralNet.predict(X)>0.5)==Y
final_accuracy = sum(((NeuralNet.predict(X)>0.5)==Y))/64
print("The network reached a success ratio of", final_accuracy,"with a learning rate of:", learning_rate)
print("optimal success ratio of", success_ratio[0], "found at epoch:", success_ratio[1])