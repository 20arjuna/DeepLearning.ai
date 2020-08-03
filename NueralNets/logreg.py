import numpy as np
from utils import *
import math

def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')

def predict(X, w, b):
    z = dot_product(X, w) + b
    yhat = sigmoid(z)

    if(yhat >= 0.5):
        return 1.0
    else:
        return 0.0

'''
Calculates the loss for a given prediction 
'''
def get_loss(yhat, y):
    loss = -1 * ( y * np.log(yhat) + (1 - y) * (np.log(1 - yhat)) )
    return loss

''' 
Calculates the derivatives
'''
def backprop(w, b, X, Y):
    #print(X)
    A = sigmoid(dot_product(w, X) + b)
    m = len(X)

    loss = -(Y*np.log(A) + (1-Y)*np.log(1-A))
    cost = (1/m) * np.sum(loss)

    dz = A - Y
    db = dz
    dw = np.zeros(len(w))

    for i in range(len(dw)):
        dw[i] = X[i] * dz

    gradients = {"dw": dw,
                 "db": db}

    return gradients, cost


'''
Updates the weights and biases

@return w the updates weights array
@return b the updated bias
'''
def gradient_descent(w, b, X, Y, iterations, alpha):
    features = len(w)
    print(X[0])
    print(type(X[0]))
    print(X[0].shape)
    for i in range(iterations):
        gradients, cost = backprop(w, b, X[i], Y[i])

        dw = gradients["dw"]
        db = gradients["db"]

        b -= alpha * db
        for i in range(features):
            w[i] -= alpha * dw[i]

        if(i%100 == 0):
            print("Cost after iteration #" + str(i) + ": " + str(cost))

    return w, b

if __name__ == '__main__':
    myFile = load_data('pima-indians-diabetes.csv')

    X_train = myFile[:384,:-1]
    #print(X_train[0])
    Y_train = myFile[:384, -1]
    #print(Y_train[0])

    X_test = myFile[384:,:-1]
    #print(X_test[0])
    Y_test = myFile[384:, -1]
    #print(Y_test[0])

    features = len(X_train[0])


    test_no = 5
    print(X_test[test_no])
    print(Y_test[test_no])

    w = np.zeros(features)
    b = 0

    print("\n" + "weights: " + str(w))
    print("bias: " + str(b))
    print("sample prediction: " + str(predict(X_test[test_no], w, b)))
    print("__________________________________________________________")

    w, b = gradient_descent(w, b, X_train, Y_train, 1000, 0.01)

    print()
    print("weights: " + str(w))
    print("bias: " + str(b))
    print("prediction: " + str(predict(X_test[test_no], w, b)))



