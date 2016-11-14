__author__ = 'rbaral'
'''
use majority voting from knn,svn,sgd and logistic regression
'''

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
    from sklearn import svm
    from sklearn import neighbors
    from sklearn.linear_model import SGDClassifier
    import numpy as np
    n_neighbors = 10
    # lets create a model
    logistic_clf = LogisticRegression(penalty='l1')
    svm_clf = svm.SVC()
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors, weights = 'uniform')
    sgd_clf = SGDClassifier(loss="hinge", penalty="l2", shuffle=True)
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

    logistic_clf.fit(trainDataList, trainLabelsList)
    svm_clf.fit(trainDataList, trainLabelsList)
    knn_clf.fit(trainDataList, trainLabelsList)
    sgd_clf.fit(trainDataList, trainLabelsList)
    #now predict the values
    print "Id,Solution"

    for index,item in enumerate(testDataList):
        prediction_one_count = 0
        prediction_zero_count = 0
        logistic_prediction = logistic_clf.predict([item])[0]
        if logistic_prediction==0:
            prediction_zero_count+=1
        else:
            prediction_one_count+=1
        knn_prediction = knn_clf.predict([item])[0]
        if knn_prediction==0:
            prediction_zero_count+=1
        else:
            prediction_one_count+=1
        '''
        sgd_prediction = sgd_clf.predict([item])[0]
        if sgd_prediction==0:
            prediction_zero_count+=1
        else:
            prediction_one_count+=1
        '''
        svm_prediction = svm_clf.predict([item])[0]
        if svm_prediction==0:
            prediction_zero_count+=1
        else:
            prediction_one_count+=1
        if prediction_zero_count>prediction_one_count:
            prediction = 0
        else:
            prediction=1
        print str(index+1)+","+str(prediction)


def performClassificationInValidation():
    from sklearn.linear_model import LogisticRegression
    from sklearn import svm
    from sklearn import neighbors
    from sklearn.linear_model import SGDClassifier
    import numpy as np
    n_neighbors = 10
    # lets create a model
    logistic_clf = LogisticRegression(penalty='l1')
    svm_clf = svm.SVC()
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors, weights = 'uniform')
    sgd_clf = SGDClassifier(loss="hinge", penalty="l2", shuffle=True)
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

    logistic_clf.fit(trainDataList, trainLabelsList)
    svm_clf.fit(trainDataList, trainLabelsList)
    knn_clf.fit(trainDataList, trainLabelsList)
    sgd_clf.fit(trainDataList, trainLabelsList)
    #now predict the values
    print "Id,Solution"
    rss = 0
    for index,item in enumerate(validationDataList):
        prediction_one_count = 0
        prediction_zero_count = 0
        logistic_prediction = logistic_clf.predict([item])[0]
        if logistic_prediction==0:
            prediction_zero_count+=1
        else:
            prediction_one_count+=1
        knn_prediction = knn_clf.predict([item])[0]
        if knn_prediction==0:
            prediction_zero_count+=1
        else:
            prediction_one_count+=1
        '''
        sgd_prediction = sgd_clf.predict([item])[0]
        if sgd_prediction==0:
            prediction_zero_count+=1
        else:
            prediction_one_count+=1
        '''
        svm_prediction = svm_clf.predict([item])[0]
        if svm_prediction==0:
            prediction_zero_count+=1
        else:
            prediction_one_count+=1
        if prediction_zero_count>prediction_one_count:
            prediction = 0
        else:
            prediction=1
        rss+=(prediction - validationDataLabels[index])**2
        print str(index+1)+","+str(prediction)
    print "rss is:",str(rss)

if __name__=="__main__":
    print "main method started"
    #performClassification()
    performClassificationInValidation()