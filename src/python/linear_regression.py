import numpy as np 
import os
import sys

#-----------------------------------------------------------------
def generateData(nRows,nColumns):

    X = np.random.normal(loc=0.0, scale=1.0, size=(nRows,nColumns))
    X[:,-1] = 1
    param = np.random.random((nColumns,1))
    y = np.dot(X,param)
    return X,y,param

#-----------------------------------------------------------------
def predict(X,theta):

    return np.dot(X,theta)

#-----------------------------------------------------------------
def lossfunction(y,h):

    MSE = np.sum((y-h)**2)
    return MSE

#-----------------------------------------------------------------
if __name__ == '__main__':

    #Create Fake Date
    nRows = 100
    nColumns = 2
    X,y,orig = generateData(nRows,nColumns)
    print(orig)

    #Calculate using closes
    closedform_param = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
    print(closedform_param)

    #Loss of Closed Form function
    h = np.dot(X,closedform_param)
    MSE = np.sum((h-y)**2)

    #Calculate Using Gradient Descent
    theta = np.random.random((X.shape[1],1))
    learning_rate = 0.01
    nIter = 1000
    for i in range(nIter):
        h = predict(X,theta)
        loss = lossfunction(y,h)
        error = h - y
        gradient = np.dot(X.T,h-y) / X.shape[0]
        theta = theta - gradient*learning_rate

        #Loss of Gradient Descent
        h = np.dot(X,theta)
        MSE = np.sum((h-y)**2)