import numpy
import scipy.special
from functools import reduce
import scipy.misc

# Neural network class definition
class NeuralNetwork:

    # Initialise the network
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # Sample from a random distribution to set initial weights
        self.weights_ih = numpy.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
        self.weights_oh = numpy.random.normal(0.0, pow(self.output_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

        # Default activation function
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.weights_oh, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        # Calculate Errors at Output
        output_errors = targets - final_outputs

        # Determine derivative of errors for each hidden->output weight
        hidden_errors = numpy.dot(self.weights_oh.T, output_errors)
        self.weights_oh += self.learning_rate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.weights_ih += self.learning_rate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.weights_ih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.weights_oh, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def __str__(self):
        return (f'nn(input_nodes={self.input_nodes},'
                f'hidden_nodes={self.hidden_nodes},'
                f'output_nodes={self.output_nodes},'
                f'learning_rate={self.learning_rate}'
                f'weights_ih={self.weights_ih}'
                f'weights_oh={self.weights_oh}')
