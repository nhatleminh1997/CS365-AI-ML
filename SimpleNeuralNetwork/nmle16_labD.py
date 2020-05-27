"""Nhat Le
CS 365 Lab D Neural Network
Build a simple neural network to simulate XOR gate"""

import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x)) 

def init_weights_biases(num_input_nodes, num_hidden_nodes, num_output_nodes):
    parameter_dictionary = {}
    # Code goes here; it should add
    parameter_dictionary['hidden_biases'] = np.zeros((num_hidden_nodes,1))
    parameter_dictionary['output_biases'] = np.zeros((num_output_nodes,1))
    parameter_dictionary['hidden_weights'] = np.random.randn(num_hidden_nodes,num_input_nodes)
    parameter_dictionary['output_weights'] = np.random.randn(num_output_nodes,num_hidden_nodes)
    return parameter_dictionary


def read_file_to_array(file_name):
    with open(file_name,'r') as data_file:
        all_lines = data_file.read().split('\n')
    header_array = np.array(all_lines[0].split('\t'),ndmin=2)
    header_array = header_array.T 
    data_array = []
    for i in all_lines[1:]:
        if len(i)>0:
            data_array.append(i.split('\t'))
    data_array = np.array(data_array,dtype = 'float64', ndmin=2).T 
    feature_array = np.array(data_array[:len(data_array)-1],ndmin=2)
    label_array = np.array(data_array[-1],ndmin=2)
    return feature_array, label_array, header_array


def forward_propagate(feature_array,param_dictionary):
    hidden_layer_values = np.dot(param_dictionary['hidden_weights'],feature_array) + param_dictionary['hidden_biases']
    hidden_layer_outputs = sigmoid(hidden_layer_values)
    output_layer_values = np.dot(param_dictionary['output_weights'],hidden_layer_outputs) + param_dictionary['output_biases']
    output_layer_outputs = sigmoid(output_layer_values)
    output_vals = {"hidden_layer_outputs": hidden_layer_outputs,
    "output_layer_outputs": output_layer_outputs}
    return output_vals

def find_loss(output_layer_outputs, labels):
    # The number of examples is the number of columns in labels
    num_examples = labels.shape[1]
    loss = (-1 / num_examples) * np.sum(np.multiply(labels, np.log(output_layer_outputs)) +
                                        np.multiply(1-labels, np.log(1-output_layer_outputs)))
    return loss

def backprop(feature_array, labels, output_vals, weights_biases_dict, verbose=False):
    if verbose:
        print()
    # We get the number of examples by looking at how many total
    # labels there are. (Each example has a label.)
    num_examples = labels.shape[1]
    
    # These are the outputs that were calculated by each
    # of our two layers of nodes that calculate outputs.
    hidden_layer_outputs = output_vals["hidden_layer_outputs"]
    output_layer_outputs = output_vals["output_layer_outputs"]

    # These are the weights of the arrows coming into our output
    # node from each of the hidden nodes. We need these to know
    # how much blame to place on each hidden node.
    output_weights = weights_biases_dict["output_weights"]

    # This is how wrong we were on each of our examples, and in
    # what direction. If we have four training examples, there
    # will be four of these.
    # This calculation works because we are using binary cross-entropy,
    # which produces a fairly simply calculation here.
    raw_error = output_layer_outputs - labels
    if verbose:
        print("raw_error", raw_error)
    
    # This is where we calculate our gradient for each of the
    # weights on arrows coming into our output.
    output_weights_gradient = np.dot(raw_error, hidden_layer_outputs.T)/num_examples
    if verbose:
        print("output_weights_gradient", output_weights_gradient)
    
    # This is our gradient on the bias. It is simply the
    # mean of our errors.
    output_bias_gradient = np.sum(raw_error, axis=1, keepdims=True)/num_examples
    if verbose:
        print("output_bias_gradient", output_bias_gradient)
    
    # We now calculate the amount of error to propegate back to our hidden nodes.
    # First, we find the dot product of our output weights and the error
    # on each of four training examples. This allows us to figure out how much,
    # for each of our training examples, each hidden node contributed to our
    # getting things wrong.
    blame_array = np.dot(output_weights.T, raw_error)
    if verbose:
        print("blame_array", blame_array)
    
    # hidden_layer_outputs is the actual values output by our hidden layer for
    # each of the four training examples. We square each of these values.
    hidden_outputs_squared = np.power(hidden_layer_outputs, 2)
    if verbose:
        print("hidden_layer_outputs", hidden_layer_outputs)
        print("hidden_outputs_squared", hidden_outputs_squared)
    
    # We now multiply our blame array by 1 minus the squares of the hidden layer's
    # outputs.
    propagated_error = np.multiply(blame_array, 1-hidden_outputs_squared)
    if verbose:
        print("propagated_error", propagated_error)
    
    # Finally, we compute the magnitude and direction in which we
    # should adjust our weights and biases for the hidden node.
    hidden_weights_gradient = np.dot(propagated_error, feature_array.T)/num_examples
    hidden_bias_gradient = np.sum(propagated_error, axis=1, keepdims=True)/num_examples
    if verbose:
        print("hidden_weights_gradient", hidden_weights_gradient)
        print("hidden_bias_gradient", hidden_bias_gradient)
    
    # A dictionary that stores all of the gradients
    # These are values that track which direction and by
    # how much each of our weights and biases should move
    gradients = {"hidden_weights_gradient": hidden_weights_gradient,
                 "hidden_bias_gradient": hidden_bias_gradient,
                 "output_weights_gradient": output_weights_gradient,
                 "output_bias_gradient": output_bias_gradient}

    return gradients

def update_weights_biases(parameter_dictionary, gradients, learning_rate):
    new_hidden_weights = parameter_dictionary['hidden_weights'] - learning_rate*gradients['hidden_weights_gradient']
    new_hidden_bias = parameter_dictionary['hidden_biases'] - learning_rate*gradients['hidden_bias_gradient']
    new_output_weights = parameter_dictionary['output_weights'] - learning_rate*gradients['output_weights_gradient']
    new_output_bias = parameter_dictionary['output_biases'] - learning_rate*gradients['output_bias_gradient']
    updated_parameters = {'hidden_weights': new_hidden_weights, 
                        'hidden_biases':new_hidden_bias,
                        'output_weights': new_output_weights,
                        'output_biases': new_output_bias}
    return updated_parameters

def model_file(file_name, num_inputs, num_hiddens, num_outputs, epochs, learning_rate):
    features, labels, headers = read_file_to_array(file_name)
    param_dictionary = init_weights_biases(num_inputs,num_hiddens,num_outputs)
    for i in range(epochs):
        output_vals = forward_propagate(features, param_dictionary)
        loss = find_loss(output_vals['output_layer_outputs'],labels)
        if loss < 0.01: 
            print('Loss after '+ str(i+1)+' epochs: '+str(loss))
            return param_dictionary
        if i%1000 == 999: # report loss every 100 epochs
            print('Loss after '+ str(i+1)+' epochs: '+str(loss))
        gradients = backprop(features,labels, output_vals,param_dictionary,False)
        param_dictionary = update_weights_biases(param_dictionary,gradients,learning_rate)
    print('Loss after '+ str(epochs)+' epochs: '+str(loss))
    return param_dictionary

# trained_dictionary = model_file('xor.txt',2,2,1,1000,0.3)

if __name__=='__main__':
    for i in range(1):
        file_name = 'xor.txt'
        num_inputs = 2
        num_hiddens = 2
        num_outputs = 1
        epochs = 100000
        learning_rate = 0.3
        trained_dictionary = model_file(file_name,num_inputs,num_hiddens,num_outputs,epochs,learning_rate)
        print(trained_dictionary)




