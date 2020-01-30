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
    y = y.T.astype(int)

    #Sanity Check
    print(X.shape)
    print(y.shape)

    #Normalize X data
    for i in range(X.shape[1]):
        x = X[:,i]
        x = x - np.mean(x)
        x = x / np.std(x)
        X[:,i] = x
    
    #Init Kmeans
    nClasses = 3
    nIter = 1000
    centers = np.random.random((nClasses,X.shape[1])) - 0.5
    
    #Assign Membership
    for m in range(nIter):
        labels = []
        for i in range(X.shape[0]):
            point = X[i,:]
            distances = []
            for j in range(nClasses):
                center = centers[j,:]   
                dis = center - point
                dis = np.sqrt(np.sum(dis**2))
                distances.append(dis)
            distances = np.array(distances)
            label = np.argmin(distances)
            labels.append(label)
        labels = np.array(labels)  
        
        #Compute New Means
        for j in range(nClasses):
            indices = np.where(labels == j)[0]
            centers[j,:] = np.mean(X[indices,:],axis=0)
        
    #Compute Accuracy
    print(labels)
    print(y)



