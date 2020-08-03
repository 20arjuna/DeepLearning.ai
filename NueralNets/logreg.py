import numpy as np
from utils import *
import math

def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')

def predict(X, w, b):
    z = dot_product(X, w) + b
    yhat = sigmoid(z)
    return yhat

'''
Calculates the loss for a given prediction 
'''
def get_loss(yhat, y):
    loss = -1 * ( y * math.log(yhat) + (1 - y) * (math.log(1 - yhat)) )
    return loss

''' 
Calculates the derivatives
'''
def backprop(X, yhat, y, features):
    dz = yhat - y
    db = dz
    dw = np.empty(features)

    for i in range(features):
        dw[i] = X[i] * dz

    return [dw, db]

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

    X_train = myFile[:384, :-1]
    Y_train = myFile[:384, -1]

    X_test = myFile[384:, :-1]
    Y_test = myFile[384:, -1]

    features = len(X_train[0])
    m = len(X_train)

    print(X_test[1])
    print(Y_test[1])

    w = np.zeros(features)
    b = np.random.random()

    print("\n" + "weights: " + str(w))
    print("bias: " + str(b))
    print("sample prediction: " + str(predict(X_test[1], w, b)))

    dw = np.zeros(features)
    db = 0
    alpha = 0.05
    for iter in range(100):
        for i in range(m):
            yhat = predict(X_train[i], w, b)
            if(yhat == Y_train[i]):
                break
            loss = get_loss(yhat, Y_train[i])
            dz = yhat - loss

            for j in range(features):
                dw[j] = X_train[i][j] * dz
            db = dz

        loss /= m
        dw /= m
        db /= m

        for i in range(features):
            w[i] -= (alpha * dw[i])

        b -= (alpha * db)
        #print("reached here")

    print()
    print("weights: " + str(w))
    print("bias: " + str(b))
    print("prediction: " + str(predict(X_test[1], w, b)))



