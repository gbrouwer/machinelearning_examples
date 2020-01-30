import sys
import numpy as np 
import os
import pandas as pd 
from tqdm import tqdm

#------------------------------------------------------------------------
def readEmail(filename):

    #Determine Label
    elements = filename.split('.')
    if (elements[-2] == 'ham'):
        label = 0
    else:
        label = 1

    #Read Email
    body = ''
    with open(filename,'r' , encoding="ISO-8859-1") as f:
        subject = f.readline().rstrip()
        subject = subject.replace('Subject: ','')
        subject = subject.replace('re : ','re: ')
        for line in f:
            body = body + ' ' + line.rstrip()

    return subject, body, label

#------------------------------------------------------------------------
def cleanSubject(mystr):

    newstr = ''
    elements = mystr.split(' ')
    for item in elements:
        if (len(item) > 1):
            newstr = newstr + ' ' + item
    newstr = newstr[1:-1]
    return newstr

#------------------------------------------------------------------------
if __name__ == '__main__':

    #Get Files
    myfiles = []
    for root, dirs, files in os.walk('/Users/gbrouwer/Datascience/machinelearning/python/data/enron/raw/', topdown=False):
        for name in files:
            if '.txt' in name:
                myfiles.append(os.path.join(root, name))

    #Extract Category and subject for all files
    data = []
    for i in tqdm(range(len(myfiles))):
        subject,body,label = readEmail(myfiles[i]) 
        subject = cleanSubject(subject) 
        data.append((subject,label,body))
    data = pd.DataFrame(data)
    data.columns = ['subject','label','body']

    #Save
    data.to_csv('/Users/gbrouwer/Datascience/machinelearning/python/data/enron/processed/enron.csv',index=False)
