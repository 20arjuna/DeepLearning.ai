import numpy as np
from utils import *
import math

def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')

def predict(X, w, b):
    z = dot_product(X, w) + b
    a = sigmoid(z)
    return a

''' 
Calculates the derivatives
'''
def backprop(X, yhat, y, features):
    loss = -1 * (y * math.log(yhat) + (1 - y) * (math.log(1 - yhat)))
    dz = yhat - y
    db = dz
    dw = np.empty(features)

    for i in range(features):
        dw[i] = X[i] * dz

    return dw, db, loss

'''
Updates the weights
'''
def gradient_descent(w, dw, alpha):
    features = len(w)
    for i in range(features):
        w[i] -= alpha * dw[i]
    return w

if __name__ == '__main__':
    myFile = load_data('pima-indians-diabetes.csv')

    X = myFile[1][:-1]
    features = len(X)
    w = np.random.rand(features)
    b = np.random.random()

    prediction = predict(X, w, b)
    print(prediction)