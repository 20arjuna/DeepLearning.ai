import numpy as np
import math

'''
Computes the sigmoid functions with the given input

@param x The input to the sigmoid function
@return  The output of the sigmoid function given the param as input
'''


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1-sigmoid(x))

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
    # Y = Y.reshape(1, len(Y))

    return X, Y


class Perceptron:
    def __init__(self, features, labels, hiddenLayerNodes):
        self.X = features
        self.Y = labels
        self.training_examples = labels.size
        self.hiddenLayerNodes = hiddenLayerNodes
        self.outputNodes = 1
        self.b1 = np.random.randn(self.hiddenLayerNodes, 1)
        self.b2 = np.random.randn(self.outputNodes, 1)
        self.w1 = np.random.rand(self.hiddenLayerNodes, len(self.X[0]))
        self.w2 = np.random.rand(self.outputNodes, self.hiddenLayerNodes)

    def forward_prop(self, inputs):
        # print(self.b2)
        # print(self.b2.shape)

        # print(self.w1.shape)
        # print(inputs.shape)

        z1 = np.dot(self.w1, inputs) + self.b1
        # print(z1)
        #print(z1.shape)
        a1 = np.zeros(z1.shape)
        # print(a1)
        # print(a1.shape)
        for i in range(len(z1)):
            for j in range(len(z1[0])):
                a1[i][0] = np.tanh(z1[i][0])
        #print(a1)
        # print(a1.T.shape)
        # print(a1.T)
        z2 = np.dot(self.w2, a1) + self.b2
        #print(self.w2)
        #print(self.w2.shape)
        a2 = sigmoid(z2)
        A = {"a1": a1.T.reshape(self.hiddenLayerNodes),
             "a2": a2}
        return A

    def get_prediction_matrix(self):
        predictions = np.zeros(self.training_examples)
        hiddenPredictions = np.zeros([self.training_examples, self.hiddenLayerNodes])

        for i in range(self.training_examples):
            #print(i)
            activations = self.forward_prop(self.X[i].reshape(len(X[i]), 1))
            hiddenPredictions[i] = activations["a1"]
            predictions[i] = activations["a2"]

        #print(hiddenPredictions.shape)
        #print(predictions.shape)
        return hiddenPredictions, predictions.reshape(1, len(predictions))

    def train(self, training_iterations, alpha):
        for i in range(training_iterations):
            print("iteration #: " + str(i))
            A1, A2 = self.get_prediction_matrix()
            print(A1.shape) #(4,2)
            print(A2.shape) #(1,4)
            print(self.X.shape) #(4,2)
            print(self.Y.shape) #(4,)
            dZ2 = A2 - self.Y
            print(dZ2.shape) #(1,4)
            dW2 = (1/self.training_examples) * np.dot(dZ2, A1)
            db2 = (1/self.training_examples) * np.sum(dZ2, axis=1, keepdims=True)

            dZ1 = np.multiply(np.dot(self.w2.T, dZ2), 1 - np.power(A1.T, 2))
            dW1 = (1 / self.training_examples) * np.dot(dZ1, X)
            db1 = (1 / self.training_examples) * np.sum(dZ1, axis=1, keepdims=True)

            # print(self.w2.shape)
            # print(dW2.shape)
            self.w1 -= alpha * dW1
            self.b1 -= alpha * db1
            self.w2 -= alpha * dW2
            self.b2 -= alpha * db2
        # print(self.w1)
        # print(self.w1.shape)
        # print(self.w2)
        # print(self.w2.shape)

    def predict(self, inputs):
        # print(inputs.shape)
        # print(self.w1.shape)
        # print(self.b1)
        a1 = np.dot(inputs, self.w1)
        for i in range(len(a1)):
            for j in range(len(a1[0])):
                a1[i][j] = np.tanh(a1[i][j])

        #print(a1.shape)
        a2 = np.dot(a1, self.w2.T)
        for r in range(len(a2)):
            for j in range(len(a2[0])):
                a2[i][j] = sigmoid(a2[i][j])

        return a2



if __name__ == '__main__':
    files = ["and.csv", "or.csv", "xor.csv"]
    print("Choose your network")
    choice = int(input("1. AND" + "\n" + "2. OR" + "\n" + "3. XOR" + "\n" + "Enter choice: "))
    myFile = load_data(files[choice-1])

    X, Y = preprocess(myFile)
    # print(X)
    # print(Y)

    # arr = np.array([0, 1])
    # arr = arr.T
    # arr = arr.reshape(2,1)

    myNeuralNet = Perceptron(X, Y, 2)
    myNeuralNet.train(1000, 0.001)
    testArray = np.array([1,0])
    testArray = testArray.reshape(1, len(testArray))

    print(myNeuralNet.predict(testArray))

    #print(myNeuralNet.predict(arr))


    # inputs = np.array([0, 1, 1])
    # hidden = 4
    # myNeuralNet = Perceptron(inputs, hidden)
    # print(myNeuralNet.predict())