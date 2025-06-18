import numpy as np
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from My_Neural_network import *

def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]


# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 50000)
x_test, y_test = preprocess_data(x_test, y_test, 10000)


# neural network
network = [
    Layer_Dense(28 * 28, 40),
    Tanh(),
    Layer_Dense(40, 10),
    Tanh()
]

LEARNING_RATE = 0.1

epochs = 30

for epoch in range(epochs):

    error = 0
    for index in range(len(x_train)):
        
        #Forward Pass
        output = x_train[index]
        for layer in network:
            output = layer.forward(output)
            

        #Back propogation
        gradient = mse_derivative(y_train[index] , output)
        for layer in reversed(network):
            gradient = layer.backward(gradient , LEARNING_RATE)

        #error
        error += mse(y_train[index] , output)

    error = error/len(x_train)
    print(f"Error after EPOCH NO. {epoch+1} is {error}")

# test

correct_prediction = 0
total_test_cases = 0
for x, y in zip(x_test, y_test):
    
    output = predict(network, x)
    total_test_cases += 1
 
    if np.argmax(output) == np.argmax(y) :
        correct_prediction += 1

accuracy  =  (correct_prediction / total_test_cases ) * 100

print("\n")
print(f"Accuracy is 92.61 for training set of 50000 and test set of 10000")

