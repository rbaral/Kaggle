__author__ = 'rbaral'

from sklearn.linear_model import SGDClassifier
#import pandas as pd
import numpy as np

'''
reads the csv files and return the
list of lines
'''
def readFile(filePath):
    with open(filePath,"r") as file:
        readLinesList = file.readlines();
    return readLinesList

def loadData():
    #trainDataLabels = pd.read_csv("data/trainLabels.csv")
    #trainData = pd.read_csv("data/train.csv")
    #testData = pd.read_csv("data/test.csv")
    #print trainData.shape[0], trainDataLabels.shape[0], testData.shape[0]
    #print trainData.iloc[0]
    trainDataLabels = readFile("data/trainLabels.csv")
    trainData = readFile("data/train.csv")
    testData = readFile("data/test.csv")
    #print len(trainData),len(trainDataLabels),len(testData)
    return trainData, trainDataLabels, testData


def performClassification():
    # lets create a model
    sgd = SGDClassifier(loss="hinge", penalty="l2")
    #get the data
    trainData, trainLabels, testData = loadData()
    #prepare the data to fit to the model
    trainDataList = []
    for index,item in enumerate(trainData):
        item = item.replace("\n","")
        trainDataList.append(np.array(item.split(","),dtype="f"))
    testDataList =  []
    for index,item in enumerate(testData):
        item = item.replace("\n","")
        testDataList.append(np.array(item.split(","),dtype="f"))
    # fit the training data and label to this  model
    trainLabelsList = []
    for item in trainLabels:
        trainLabelsList.append(int(item.replace("\n","")))
    sgd.fit(trainDataList, trainLabelsList)
    #now predict the values
    print "Id,Solution"
    for index,item in enumerate(testDataList):
        print str(index+1)+","+str(sgd.predict([item])[0])


def performClassificationInValidation():
    # lets create a model
    sgd = SGDClassifier(loss="hinge", penalty="l2", shuffle=True)
    #get the data
    trainData, trainLabels, testData = loadData()
    #prepare the data to fit to the model
    trainDataList = []
    for index,item in enumerate(trainData):
        item = item.replace("\n","")
        trainDataList.append(np.array(item.split(","),dtype="f"))
    testDataList =  []
    for index,item in enumerate(testData):
        item = item.replace("\n","")
        testDataList.append(np.array(item.split(","),dtype="f"))
    # fit the training data and label to this  model
    trainLabelsList = []
    for item in trainLabels:
        trainLabelsList.append(int(item.replace("\n","")))

    #now get the validation set from 10% of the trainDataList
    validationListRange = int(len(trainDataList)-0.1*len(trainDataList))
    validationDataList = trainDataList[validationListRange:]
    validationDataLabels = trainLabelsList[validationListRange:]

    trainDataList = trainDataList[:validationListRange]
    trainLabelsList = trainLabelsList[:validationListRange]

    sgd.fit(trainDataList, trainLabelsList)
    #now predict the values
    print "Id,Solution"
    rss = 0
    for index,item in enumerate(validationDataList):
        prediction = sgd.predict([item])[0]
        rss+=(prediction - validationDataLabels[index])**2
        print str(index+1)+","+str(prediction)
    print "rss is:",str(rss)

if __name__=="__main__":
    print "main method started"
    #performClassification()
    performClassificationInValidation()