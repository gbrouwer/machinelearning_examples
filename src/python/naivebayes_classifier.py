import sys
import numpy as np 
import os
import pandas as pd 
import pickle
from tqdm import tqdm

#------------------------------------------------------------------------
def classifySubject(mystr,S,H):

    #Classify
    label = 0
    elements = mystr.split(' ')
    spamProp = 0.0
    hamProp = 0.0
    spamPropNorm = 0.0
    hamPropNorm = 0.0
    for word in elements:
        if word in S:
            spamProp = spamProp + np.log(S[word])
            hamProp = hamProp + np.log(H[word])
    spamProp = np.exp(spamProp)
    hamProp = np.exp(hamProp)
    if (spamProp > 0.0 and hamProp > 0.0):
        spamPropNorm = spamProp / (spamProp + hamProp)
        hamPropNorm = hamProp / (spamProp + hamProp)
        if (spamPropNorm > hamPropNorm):
            label = 1
        if (spamPropNorm < hamPropNorm):
            label = 0

    #Return
    return spamPropNorm,hamPropNorm,label

#------------------------------------------------------------------------
def loadDics():

    D = pickle.load( open( "/Users/gbrouwer/Datascience/machinelearning/python/data/enron/pickle/enron.p","rb"))
    return D

#------------------------------------------------------------------------
if __name__ == '__main__':

    #Load Dictionaries
    D = loadDics()
    S = D['S']
    H = D['H']

    #Load Emails
    data = pd.read_csv('/Users/gbrouwer/Datascience/machinelearning/python/data/enron/processed/enron.csv')

    #GrabD Data
    subjects = data['subject'].values
    labels = data['label'].values

    #Loop Through Subjects
    pred = []
    for i in tqdm(range(len(subjects))):
        subject = str(subjects[i])
        spamPropNorm,hamPropNorm,label = classifySubject(subject,S,H)
        pred.append(label)
    pred = np.array(pred)

    #Accuracy
    accuracy = float(np.sum(pred == labels)) / pred.shape[0]
    print(accuracy)