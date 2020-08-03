import math

'''
Computes the dot product of two arrays

@param arr1 The first array
@param arr2 The second array
@return     The dot product of the two params
@pre        Both arrays must be the same length
'''

def dot_product(arr1, arr2):
    length = len(arr1)
    ans = 0
    for i in range(length):
        ans += (arr1[i] * arr2[i])
    return ans

'''
Computes the sigmoid functions with the given input

@param x The input to the sigmoid function
@return  The output of the sigmoid function given the param as input
'''
def sigmoid(x):
    return 1 / (1 + math.exp(-x))