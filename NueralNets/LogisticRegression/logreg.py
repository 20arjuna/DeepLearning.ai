import numpy as np
from NueralNets.utils import sigmoid


def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')

'''
Makes a prediction given some inputs, weights, and bias

@param X 
        The inputs dim=(features,)
@param w 
        The weights dim=(features,)
@param b 
        The bias

@return The prediction
'''
def predict(X, w, b):
    z = np.dot(w, X) + b
    yhat = sigmoid(z)

    # if(yhat >= 0.5):
    #     return 1.0
    # else:
    #     return 0.0
    return yhat

''' 
Calculates the partial derivatives of loss with respect 
to weights and bias. (dw, db)

@param w 
        The weights array. 
        Has the same length as there are features
        dim=(features,)
@param b 
        The bias
@param X 
        The input array                 
        dim=(features,)
@param Y 
        The expected output
@return gradients
         A dictionary containing (dw,db)
         The gradients with respect to weights and bias
@return cost
         The loss function averaged over the number of training examples
'''
def backprop(w, b, X, Y):
    A = sigmoid(np.dot(w, X) + b)
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

@param w 
        The weights array
        dim=(features,)
@param b
        The bias
@param X
        The input array
        dim=(features,m)
@param Y
        The expected output array
        dim=(m,)
@param iterations
        The number of training iterations
@param alpha
        The learning rate
@return w 
         the updated weights array
@return b 
         the updated bias
'''
def gradient_descent(w, b, X, Y, iterations, alpha):
    features = len(w)
    #print(X.shape)
    for i in range(iterations):
        for k in range(X.shape[0]):
            #print("iteration #: " + str(i))
            gradients, cost = backprop(w, b, X[k], Y[k])

            dw = gradients["dw"]
            db = gradients["db"]

            b -= alpha * db
            for j in range(features):
                w[j] -= alpha * dw[j]

        print("Cost after iteration #" + str(i) + ": " + str(cost))

    return w, b

if __name__ == '__main__':
    myFile = load_data('pima-indians-diabetes.csv')

    X_train = myFile[:384,:-1]
    Y_train = myFile[:384, -1]

    X_test = myFile[384:,:-1]
    Y_test = myFile[384:, -1]

    features = len(X_train[0])

    test_no = 2
    print(X_test[test_no])
    print(Y_test[test_no])

    w = np.random.randn(features)
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



