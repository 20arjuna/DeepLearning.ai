import numpy as np
import math

'''
Computes the sigmoid functions with the given input

@param x The input to the sigmoid function
@return  The output of the sigmoid function given the param as input
'''
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

class Perceptron:
    def __init__(self, features, hiddenLayerNodes):
        self.features = features.reshape(len(features),1)
        self.hiddenLayerNodes = hiddenLayerNodes
        self.outputNodes = 1
        self.bias = np.random.randn(hiddenLayerNodes, 1)
        self.w1 = np.random.randn(hiddenLayerNodes, features)

    def predict(self):
        z1 = np.dot(self.weights[0], self.features) + self.bias
        a1 = sigmoid(z1)
        z2 =


if __name__ == '__main__':
