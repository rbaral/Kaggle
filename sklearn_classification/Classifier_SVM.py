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
    from sklearn import svm
    import numpy as np
    
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
    #custom_weight = len(np.where(np.array(trainLabelsList)==1)[0])/float(len(trainLabelsList))
    #print custom_weight
    # lets create a model
    #clf = svm.SVC(C=256, gamma=0.015625, class_weight={0: 0.51, 1: 0.49})
    clf = svm.SVC(C=256, gamma=0.015625)
    
    clf.fit(trainDataList, trainLabelsList)

    #now predict the values
    print "Id,Solution"
    for index,item in enumerate(testDataList):
        print str(index+1)+","+str(clf.predict([item])[0])

def performCrossValidation():
    from sklearn import svm
    import numpy as np
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
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

    # Cross-validation for SVM
    useSVM = True
    useCV = True

    if useSVM:
        n_folds = 10
        cv = StratifiedKFold(n_folds)
        Cs = np.power(2, np.arange(8.0, 14.0))
        kernels = ['rbf']
        gammas = np.power(2, np.arange(-7.0, -2.0))

        svc = svm.SVC()
        if useCV:
            gscv = GridSearchCV(estimator=svc, param_grid=dict(C=Cs, kernel=kernels, gamma=gammas),
                            n_jobs=1, cv=list(cv.split(trainDataList, trainLabelsList)), verbose=2)
            gscv.fit(trainDataList, trainLabelsList)
            best_params = gscv.best_params_
        else:
            best_params = {'gamma': 0.015625, 'C': 8192.0, 'kernel': 'rbf'}
        print best_params

def performClassificationInValidation():
    from sklearn import svm
    import numpy as np
    # lets create a model
    clf = svm.SVC(kernel='rbf')
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
    performClassification()
    #performClassificationInValidation()
    #performCrossValidation()