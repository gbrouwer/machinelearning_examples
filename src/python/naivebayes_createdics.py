import sys
import numpy as np 
import os
import pandas as pd 
import pickle
from tqdm import tqdm

#------------------------------------------------------------------------
def loadStopwords():

    stopwords = []
    with open('/Users/gbrouwer/Datascience/machinelearning/python/data/stopwords','r') as f:
        for line in f:
            stopwords.append(line.rstrip())

    return stopwords

#------------------------------------------------------------------------
def addToDict(D,total,mystr,stopwords):

    elements = mystr.split(' ')
    for item in elements:
        if item not in stopwords:
            if item in D:
                values = D[item]
                D[item] = values + 1
            else:
                D[item] = 1
            total = total + 1

    return(D,total)

#------------------------------------------------------------------------
if __name__ == '__main__':

    #Load
    data = pd.read_csv('/Users/gbrouwer/Datascience/machinelearning/python/data/enron/processed/enron.csv')

    #Build Spam/Ham Dictionary from the subjects
    subjects = data['subject'].values
    labels = data['label'].values

    #Load Stopwords
    stopwords = loadStopwords()

    #Dictionaries
    S = {}
    H = {}
    Stotal = 0
    Htotal = 0
    for i in tqdm(range(len(subjects))):
        subject = str(subjects[i])
        label = labels[i]
        if (label == 0):
            H,Htotal = addToDict(H,Htotal,subject,stopwords)
        if (label == 1):
            S,Stotal = addToDict(S,Stotal,subject,stopwords)            

    #Match words between dictionaries
    for item in S:
        if (item not in H):
            H[item] = 1
            Htotal = Htotal + 1
    for item in H:
        if (item not in S):
            S[item] = 1
            Stotal = Stotal + 1

    #Normalize
    for item in H:
        value = H[item]
        value = value / Htotal
        H[item] = value
    for item in S:
        value = S[item]
        value = value / Stotal
        S[item] = value

    D = {}
    D['S'] = S
    D['H'] = H
    D['Stotal'] = Stotal
    D['Htotal'] = Htotal
    pickle.dump(D, open("/Users/gbrouwer/Datascience/machinelearning/python/data/enron/pickle/enron.p","wb"))

