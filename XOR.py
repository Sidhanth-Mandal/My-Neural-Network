from My_Neural_network import *
import numpy as np

X = np.reshape([[0,0],[1,0],[0,1],[1,1]] , (4,2,1))
y = np.reshape([[1],[0],[0],[1]], (4,1,1))

LEARNING_RATE = 0.1


network = [Layer_Dense(2,3) , Tanh() , Layer_Dense(3,1) , Tanh()]

epochs = 500

for epoch in range(epochs):

    error = 0
    for index in range(len(X)):
        
        #Forward Pass
        output = X[index]
        for layer in network:
            output = layer.forward(output)
            

        #Back propogation
        gradient = mse_derivative(y[index] , output)
        for layer in reversed(network):
            gradient = layer.backward(gradient , LEARNING_RATE)

        #error
        error += mse(y[index] , output)

    error = error/len(X)
    print(f"Error after EPOCH NO. {epoch+1} is {error}")

output = [[1],[1]]
for layer in network:
    output = layer.forward(output)

print(output)



