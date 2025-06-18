import numpy as np

class Layer_Dense:

    def __init__(self , n_inputs , n_neurons):
        
        # randn will make a matrix of n_inputs X n_neurons with random values
        self.weights = np.random.randn(n_neurons ,n_inputs )
        self.biases = np.zeros(( n_neurons , 1))
        self.input = None

    def forward(self, inputs):
        self.input = inputs
        self.output = np.dot(self.weights , self.input) + self.biases

        return self.output
    
    def backward(self , output_gradient , learning_rate ):

        input_gradient = np.dot(self.weights.T , output_gradient) 

        weight_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * weight_gradient

        self.biases -= learning_rate * output_gradient
        
        return input_gradient
        
        
class Activation:

    def __init__(self , activation , activation_derivative):
        
        self.activation = activation
        self.activation_derivative = activation_derivative
        self.input = None

    def forward(self , inputs):
        
        self.input = inputs
        return self.activation(self.input)
        

    def backward(self , output_gradient , learning_rate):

        output = np.multiply(output_gradient , self.activation_derivative(self.input))
        return output

class Tanh(Activation):
    
    def __init__(self):
        
        tanh = lambda x : np.tanh(x)
        tanh_derivative = lambda x : 1 - np.tanh(x)**2
        super().__init__(tanh , tanh_derivative)  
        
        ''' this super().__init__  will pass given argument in it I.E tanh , tanh_derivative to its parent class __init__
            that is    activation --> tanh
                       activation_derivative --> tanh_derivative '''
        
def mse(y_true , y_pred):
    return np.mean(np.power(y_true - y_pred , 2))

def mse_derivative(y_true , y_pred):
    return 2 * (y_pred - y_true) / np.size(y_true)

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output