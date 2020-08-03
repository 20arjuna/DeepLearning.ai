import numpy as np
from utils import *

def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')

def predict(X, w, b):
    z = dot_product(X, w) + b
    a = sigmoid(z)
    return a

if __name__ == '__main__':
    myFile = load_data('pima-indians-diabetes.csv')

    X = myFile[1][:-1]
    features = len(X)
    w = np.random.rand(features)
    b = np.random.random()

    prediction = predict(X, w, b)
    print(prediction)