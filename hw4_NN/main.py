'''
source: https://www.analyticsvidhya.com/blog/2017/05/neural-network-from-scratch-in-python-and-r/
'''
import InputAndLabels
import numpy as np

X, Y = InputAndLabels.getValuesAndLabels()
X = np.array(X)
#Xtest = X[50:]
#X = X[:50]

Y = [[Y[i]] for i in range(len(Y))]
Y = np.array(Y)
#Ytest = Y[50:]
#Y = Y[:50]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoidDerivative(x):
    return x*(1-x)

# variable initialization
epoch = 1000
Learning_Rate = 0.01
input_layer_neurons = X.shape[1]
hidden_layer_neurons = 2
output_neurons = 1

weight_hidden = np.random.uniform(size=(input_layer_neurons,hidden_layer_neurons))
bias_hidden=np.random.uniform(size=(1,hidden_layer_neurons))
weight_out=np.random.uniform(size=(hidden_layer_neurons,output_neurons))
bias_output = np.random.uniform(size=(1,output_neurons))

optimal_success_ratio = 0
epoch_number = 0

for i in range(epoch):
    #forward prop
    hidden_layer_input1=np.dot(X,weight_hidden)
    hidden_layer_input=hidden_layer_input1 + bias_hidden
    hiddenlayer_activations = sigmoid(hidden_layer_input)
    output_layer_input1=np.dot(hiddenlayer_activations,weight_out)
    output_layer_input= output_layer_input1+ bias_output
    output = sigmoid(output_layer_input)

    #backwardProp
    E = Y-output
    slope_output_layer = sigmoidDerivative(output)
    slope_hidden_layer = sigmoidDerivative(hiddenlayer_activations)
    d_output = E * slope_output_layer
    Error_at_hidden_layer = d_output.dot(weight_out.T)
    d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    weight_out += hiddenlayer_activations.T.dot(d_output) * Learning_Rate
    bias_output += np.sum(d_output, axis=0,keepdims=True) * Learning_Rate
    weight_hidden += X.T.dot(d_hiddenlayer) * Learning_Rate
    bias_hidden += np.sum(d_hiddenlayer, axis=0,keepdims=True) * Learning_Rate

    if sum((output>0.5)==Y)/64 > optimal_success_ratio:
        optimal_success_ratio = sum((output>0.5)==Y)/64
        epoch_number = i

        print("The optimal success ratio is:", optimal_success_ratio, "and it occurs on epoch:", epoch_number)

output_thresholded = output>0.5
print(Y==output_thresholded)

print("--- now testing ---")
'''
hidden_layer_input1=np.dot(Xtest,weight_hidden)
hidden_layer_input=hidden_layer_input1 + bias_hidden
hiddenlayer_activations = sigmoid(hidden_layer_input)
output_layer_input1=np.dot(hiddenlayer_activations,weight_out)
output_layer_input= output_layer_input1+ bias_output
output = sigmoid(output_layer_input)
output_thresholded = output>0.5
print(Ytest==output_thresholded)
'''