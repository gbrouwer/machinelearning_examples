import os
import sys
import numpy as np 
import pandas as pd 


#------------------------------------------------------------
def predict(X,theta):

    return np.dot(X,theta)

#------------------------------------------------------------
def sigmoid(z):

    return 1 / (1 + np.exp(-z))

#------------------------------------------------------------
def logloss(y,h):

    loss = -y * np.log(h) + (1-y) * np.log(1-h)
    return loss

#------------------------------------------------------------
def threshold(h):

    return (h > 0.5) * 1.0

#------------------------------------------------------------
if __name__ == '__main__':

    #Read Data
    data = pd.read_csv('machinelearning/python/data/iris.data')
    
    #Remove One Class
    data = data[data['class'] != 'Iris-virginica']

    #Seperate X from y
    columns = data.columns
    X = data[columns[:4]].values
    classes = data[columns[4:]].values

    #Y to numeric
    classNames = np.unique(classes)
    y = np.zeros((classes.shape[0],1))
    for c,className in enumerate(classNames):
        print(c,className)
        indices = np.where(classes == className)[0]
        y[indices] = c
    
    #Sanity Check
    print(X.shape)
    print(y.shape)

    #Add Intercept Column to X  
    intercept = np.zeros((X.shape[0],1))
    X = np.hstack((X,intercept))

    #Parameters
    learningRate = 0.0001
    nIter = 100

    #Init
    theta = np.random.random((X.shape[1],1))
    
    #Prediction
    lossCurve = []
    accuracies = []
    for i in range(nIter):

        #Run Gradient Descent
        z = predict(X,theta)
        h = sigmoid(z) 
        loss = logloss(y,h)
        lossCurve.append(loss)
        gradient = np.dot(X.T,h-y)
        theta = theta - gradient*learningRate
        
        #Compute Accuracy
        c = threshold(h)
        accuracy = np.sum((c==y)*1) / X.shape[0]
        accuracies.append(accuracy)
    
    print(accuracies[-1])