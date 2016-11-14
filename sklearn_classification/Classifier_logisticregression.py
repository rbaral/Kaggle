__author__ = 'rbaral'

'''
reads the csv files and return the
list of lines
'''
def readFile(filePath):
    with open(filePath,"r") as file:
        readLinesList = file.readlines();
    return readLinesList

def loadData():
    trainDataLabels = readFile("data/trainLabels.csv")
    trainData = readFile("data/train.csv")
    testData = readFile("data/test.csv")
    #print len(trainData),len(trainDataLabels),len(testData)
    return trainData, trainDataLabels, testData


def performClassification():
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    # lets create a model
    clf = LogisticRegression(penalty='l1')
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
    clf.fit(trainDataList, trainLabelsList)

    #now predict the values
    print "Id,Solution"
    for index,item in enumerate(testDataList):
        print str(index+1)+","+str(clf.predict([item])[0])



def performClassificationInValidation():
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    # lets create a model
    clf = LogisticRegression(penalty='l1')
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

    clf.fit(trainDataList, trainLabelsList)
    #now predict the values
    print "Id,Solution"
    rss = 0
    for index,item in enumerate(validationDataList):
        prediction = clf.predict([item])[0]
        rss+=(prediction - validationDataLabels[index])**2
        print str(index+1)+","+str(prediction)
    print "rss is:",str(rss)

if __name__=="__main__":
    print "main method started"
    #performClassification()
    performClassificationInValidation()