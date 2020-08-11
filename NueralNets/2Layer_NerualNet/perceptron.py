import numpy as np
import math

'''
Computes the sigmoid functions with the given input

@param x The input to the sigmoid function
@return  The output of the sigmoid function given the param as input
'''


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


'''
Takes in a csv file and loads it into a numpy array

@param filename The filepath of the csv file
@return         A numpy array with the data
'''


def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')


'''
Converts the data into an input (X) and output (Y) numpy 
array. The input (X) is of dimensions (#inputs, #training examples)
and the output (Y) is of dimensions (#outputs, #training examples)

@param data The numpy array with the raw data (unprocessed)
@return X   The input numpy array with dim = (#inputs, #training examples)
@return Y   The output numpy array with dim = (#outpus, #training examples)
'''


def preprocess(data):
    X = []
    Y = []

    for training_example in data:
        X.append(training_example[:-1].T)
        Y.append(training_example[-1])

    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    Y = Y.reshape(1, len(Y))

    return X.T, Y


class Perceptron:
    def __init__(self, features, hiddenLayerNodes):
        self.features = features.reshape(len(features),1)
        self.hiddenLayerNodes = hiddenLayerNodes
        self.outputNodes = 1
        self.b1 = np.random.randn(self.hiddenLayerNodes, 1)
        self.b2 = np.random.randn(self.outputNodes, 1)
        self.w1 = np.random.randn(self.hiddenLayerNodes, len(features))
        self.w2 = np.random.randn(self.outputNodes, self.hiddenLayerNodes)

    def predict(self):
        print(self.b2)
        print(self.b2.shape)
        z1 = np.dot(self.w1, self.features) + self.b1
        print(z1)
        print(z1.shape)
        a1 = np.zeros(z1.shape)
        print(a1)
        print(a1.shape)
        for i in range(len(z1)):
            for j in range(len(z1[0])):
                a1[i][0] = sigmoid(z1[i][0])
        z2 = np.dot(self.w2, a1) + self.b2
        print(z2)
        print(z2.shape)
        a2 = sigmoid(z2)
        return a2


if __name__ == '__main__':
    files = ["and.csv", "or.csv", "xor.csv"]
    print("Choose your network")
    choice = int(input("1. AND" + "\n" + "2. OR" + "\n" + "3. XOR" + "\n" + "Enter choice: "))
    myFile = load_data(files[choice-1])

    X, Y = preprocess(myFile)
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    print()
    print(X)
    print(Y)


    # inputs = np.array([0, 1, 1])
    # hidden = 4
    # myNeuralNet = Perceptron(inputs, hidden)
    # print(myNeuralNet.predict())