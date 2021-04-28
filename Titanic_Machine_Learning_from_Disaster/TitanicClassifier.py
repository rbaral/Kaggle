__author__ = 'rbaral'
'''
In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive.
In particular, we ask you to apply the tools of machine learning to predict which passengers survived
the tragedy.
Ref:https://www.kaggle.com/davidfumo/titanic/titanic-machine-learning-from-a-disaster
'''

# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
#%matplotlib inline


def loadData():
    # get titanic train & test csv files as a DataFrame
    train_df = pd.read_csv("data/train.csv", dtype={"Age": np.float64}, )
    test_df    = pd.read_csv("data/test.csv", dtype={"Age": np.float64}, )

    # preview the data
    #print train_df.head()
    #print("----------------------------")
    #print test_df.info()
    # drop unnecessary columns, these columns won't be useful in analysis and prediction
    train_df = train_df.drop(['PassengerId','Name','Ticket'], axis=1)
    test_df    = test_df.drop(['Name','Ticket'], axis=1)
    # only in train_df, fill the two missing values with the most occurred value, which is "S".
    train_df["Embarked"] = train_df["Embarked"].fillna("S")

    # plot
    #sns.factorplot('Embarked','Survived', data=train_df, size=4,aspect=3)
    #sns.plt.show()
    #print train_df['Embarked']
    #print train_df['Survived']

    #create the subplots of 1x3 with the given figure sizes

    #fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))

    #create a plot with the count of different Embarked values

    #sns.countplot(x='Embarked', data=train_df, ax=axis1)

    #create a plot with the count of different Survived values based on(grouped by) the Embarked values

    #sns.countplot(x='Survived', hue="Embarked", data=train_df, order=[1,0], ax=axis2)

    # group by embarked, and get the mean for survived passengers for each value in Embarked
    #embark_perc = train_df[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean()

    #sns.barplot(x='Embarked', y='Survived', data=embark_perc,order=['S','C','Q'],ax=axis3)
    #sns.plt.show()

    return train_df, test_df


def insertDummyForEmbark(train_df, test_df):
    # Either to consider Embarked column in predictions,
    # and remove "S" dummy variable,
    # and leave "C" & "Q", since they seem to have a good rate for Survival.

    # OR, don't create dummy variables for Embarked column, just drop it,
    # because logically, Embarked doesn't seem to be useful in prediction.

    embark_dummies_titanic  = pd.get_dummies(train_df['Embarked'])
    #print train_df['Embarked'][:10]
    #print embark_dummies_titanic.head(n=10)
    embark_dummies_titanic.drop(['S'], axis=1, inplace=True)

    embark_dummies_test  = pd.get_dummies(test_df['Embarked'])
    embark_dummies_test.drop(['S'], axis=1, inplace=True)

    train_df = train_df.join(embark_dummies_titanic)
    test_df    = test_df.join(embark_dummies_test)

    train_df.drop(['Embarked'], axis=1,inplace=True)
    test_df.drop(['Embarked'], axis=1,inplace=True)

    return train_df, test_df

def preProcessingForFare(train_df, test_df):
    # only for test_df, since there is a missing "Fare" values
    test_df["Fare"].fillna(test_df["Fare"].median(), inplace=True)

    # convert from float to int
    train_df['Fare'] = train_df['Fare'].astype(int)
    test_df['Fare']    = test_df['Fare'].astype(int)

    # get fare for survived & didn't survive passengers
    fare_not_survived = train_df[train_df["Survived"] == 0]["Fare"]
    fare_survived     = train_df[train_df["Survived"] == 1]["Fare"]

    # get average and std for fare of survived/not survived passengers
    avgerage_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
    std_fare      = DataFrame([fare_not_survived.std(), fare_survived.std()])

    # plot
    #train_df['Fare'].plot(kind='hist', figsize=(15,3), bins=100, xlim=(0,50))

    #avgerage_fare.index.names = std_fare.index.names = ["Survived"]
    #avgerage_fare.plot(yerr=std_fare,kind='bar',legend=False)
    #plt.show()
    return train_df, test_df

def preProcessingForAge(train_df, test_df):
    fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,4))
    axis1.set_title('Original Age values - Titanic')
    axis2.set_title('New Age values - Titanic')
    
    # axis3.set_title('Original Age values - Test')
    # axis4.set_title('New Age values - Test')
    
    # get average, std, and number of NaN values in train_df
    average_age_titanic   = train_df["Age"].mean()
    std_age_titanic       = train_df["Age"].std()
    #get count by subtracting the non-null rows from the total rows
    count_nan_age_titanic = len(train_df["Age"])-train_df["Age"].count() #train_df["Age"].isnull().sum()
    #print train_df["Age"].isnull().count(), train_df["Age"].isnull().sum()
    # get average, std, and number of NaN values in test_df
    average_age_test   = test_df["Age"].mean()
    std_age_test       = test_df["Age"].std()
    #get count by subtracting the non-null rows from the total rows
    count_nan_age_test = len(test_df["Age"])-test_df["Age"].count()#test_df["Age"].isnull().sum()
    
    # generate random numbers between (mean - std) & (mean + std)
    rand_1 = np.random.randint(average_age_titanic - std_age_titanic, average_age_titanic + std_age_titanic, size = count_nan_age_titanic)
    rand_2 = np.random.randint(average_age_test - std_age_test, average_age_test + std_age_test, size = count_nan_age_test)
    
    # plot original Age values
    # NOTE: drop all null values, and convert to int
    train_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)
    # test_df['Age'].dropna().astype(int).hist(bins=70, ax=axis1)
    
    # fill NaN values in Age column with random values generated
    train_df["Age"][np.isnan(train_df["Age"])] = rand_1
    test_df["Age"][np.isnan(test_df["Age"])] = rand_2

    # convert from float to int
    train_df['Age'] = train_df['Age'].astype(int)
    test_df['Age']    = test_df['Age'].astype(int)
            
    # plot new Age Values
    train_df['Age'].hist(bins=70, ax=axis2)
    # test_df['Age'].hist(bins=70, ax=axis4)
    #plt.show()

    # peaks for survived/not survived passengers by their age
    facet = sns.FacetGrid(train_df, hue="Survived",aspect=4)
    facet.map(sns.kdeplot,'Age',shade= True)
    facet.set(xlim=(0, train_df['Age'].max()))
    facet.add_legend()
    # average survived passengers by age
    fig, axis1 = plt.subplots(1,1,figsize=(18,4))
    average_age = train_df[["Age", "Survived"]].groupby(['Age'],as_index=False).mean()
    sns.barplot(x='Age', y='Survived', data=average_age)
    #plt.show()

    return train_df, test_df

def preProcessingForCabin(train_df, test_df):
    # It has a lot of NaN values, so it won't cause a remarkable impact on prediction
    train_df.drop("Cabin",axis=1,inplace=True)
    test_df.drop("Cabin",axis=1,inplace=True)
    return train_df, test_df


def preProcessingForFamily(train_df, test_df):
    # Instead of having two columns Parch & SibSp,
    # we can have only one column represent if the passenger had any family member aboard or not,
    # Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.
    train_df['Family'] =  train_df["Parch"] + train_df["SibSp"]
    train_df['Family'].loc[train_df['Family'] > 0] = 1
    train_df['Family'].loc[train_df['Family'] == 0] = 0

    test_df['Family'] =  test_df["Parch"] + test_df["SibSp"]
    test_df['Family'].loc[test_df['Family'] > 0] = 1
    test_df['Family'].loc[test_df['Family'] == 0] = 0

    # drop Parch & SibSp
    train_df = train_df.drop(['SibSp','Parch'], axis=1)
    test_df    = test_df.drop(['SibSp','Parch'], axis=1)

    # plot
    fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(10,5))

    # sns.factorplot('Family',data=train_df,kind='count',ax=axis1)
    sns.countplot(x='Family', data=train_df, order=[1,0], ax=axis1)

    # average of survived for those who had/didn't have any family member
    family_perc = train_df[["Family", "Survived"]].groupby(['Family'],as_index=False).mean()
    sns.barplot(x='Family', y='Survived', data=family_perc, order=[1,0], ax=axis2)

    axis1.set_xticklabels(["With Family","Alone"], rotation=0)

    return train_df, test_df

def get_person(passenger):
    age,sex = passenger
    return 'child' if age < 16 else sex

def preProcessingForGender(train_df, test_df):
    # As we see, children(age < ~16) on aboard seem to have a high chances for Survival.
    # So, we can classify passengers as males, females, and child
    train_df['Person'] = train_df[['Age','Sex']].apply(get_person,axis=1)
    test_df['Person']    = test_df[['Age','Sex']].apply(get_person,axis=1)

    # No need to use Sex column since we created Person column
    train_df.drop(['Sex'],axis=1,inplace=True)
    test_df.drop(['Sex'],axis=1,inplace=True)

    # create dummy variables for Person column, & drop Male as it has the lowest average of survived passengers
    person_dummies_titanic  = pd.get_dummies(train_df['Person'])
    person_dummies_titanic.columns = ['Child','Female','Male']
    person_dummies_titanic.drop(['Male'], axis=1, inplace=True)

    person_dummies_test  = pd.get_dummies(test_df['Person'])
    person_dummies_test.columns = ['Child','Female','Male']
    person_dummies_test.drop(['Male'], axis=1, inplace=True)

    train_df = train_df.join(person_dummies_titanic)
    test_df    = test_df.join(person_dummies_test)

    fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))

    # sns.factorplot('Person',data=train_df,kind='count',ax=axis1)
    sns.countplot(x='Person', data=train_df, ax=axis1)

    # average of survived for each Person(male, female, or child)
    person_perc = train_df[["Person", "Survived"]].groupby(['Person'],as_index=False).mean()
    sns.barplot(x='Person', y='Survived', data=person_perc, ax=axis2, order=['male','female','child'])

    train_df.drop(['Person'],axis=1,inplace=True)
    test_df.drop(['Person'],axis=1,inplace=True)
    
    return train_df, test_df

def preProcessingForPclass(train_df, test_df):
    # sns.factorplot('Pclass',data=train_df,kind='count',order=[1,2,3])
    sns.factorplot('Pclass','Survived',order=[1,2,3], data=train_df,size=5)
    
    # create dummy variables for Pclass column, & drop 3rd class as it has the lowest average of survived passengers
    pclass_dummies_titanic  = pd.get_dummies(train_df['Pclass'])
    pclass_dummies_titanic.columns = ['Class_1','Class_2','Class_3']
    pclass_dummies_titanic.drop(['Class_3'], axis=1, inplace=True)
    
    pclass_dummies_test  = pd.get_dummies(test_df['Pclass'])
    pclass_dummies_test.columns = ['Class_1','Class_2','Class_3']
    pclass_dummies_test.drop(['Class_3'], axis=1, inplace=True)
    
    train_df.drop(['Pclass'],axis=1,inplace=True)
    test_df.drop(['Pclass'],axis=1,inplace=True)
    
    train_df = train_df.join(pclass_dummies_titanic)
    test_df    = test_df.join(pclass_dummies_test)
    #sns.plt.show()
    return train_df, test_df


def predictWithLogisticRegression(train_features, train_lables, test_features):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(train_features, train_lables)
    print "PassengerId,Survived"

    for index in range(len(test_features)):
        print str(test_features.iloc[index]["PassengerId"].astype(int))+","+str(clf.predict(test_features.iloc[index].drop("PassengerId"))[0])

    print clf.score(train_features, train_lables)


def predictionWithSVM(train_features, train_lables, test_features):
    from sklearn.svm import SVC, LinearSVC

    svc = SVC()
    svc.fit(train_features, train_lables)
    print "PassengerId,Survived"

    for index in range(len(test_features)):
        print str(test_features.iloc[index]["PassengerId"].astype(int))+","+str(svc.predict(test_features.iloc[index].drop("PassengerId"))[0])

    print svc.score(train_features, train_lables)

    #using LinearSVC
    '''
    svc = LinearSVC()
    svc.fit(train_features, train_lables)
    print "PassengerId,Survived"

    for index in range(len(test_features)):
        print str(test_features.iloc[index]["PassengerId"].astype(int))+","+str(svc.predict(test_features.iloc[index].drop("PassengerId"))[0])

    print svc.score(train_features, train_lables)
    '''

def predictionWithRandomForest(train_features, train_labels, test_features):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_features, train_labels)
    print "PassengerId,Survived"

    for index in range(len(test_features)):
        print str(test_features.iloc[index]["PassengerId"].astype(int))+","+str(clf.predict(test_features.iloc[index].drop("PassengerId"))[0])

    print clf.score(train_features, train_labels)

def predictionWithGaussianNB(train_features, train_labels, test_features):
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
    clf.fit(train_features, train_lables)
    print "PassengerId,Survived"

    for index in range(len(test_features)):
        print str(test_features.iloc[index]["PassengerId"].astype(int))+","+str(clf.predict(test_features.iloc[index].drop("PassengerId"))[0])

    print clf.score(train_features, train_labels)

def predictionWithKNN(train_features, train_labels, test_features):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(train_features, train_lables)
    print "PassengerId,Survived"

    for index in range(len(test_features)):
        print str(test_features.iloc[index]["PassengerId"].astype(int))+","+str(clf.predict(test_features.iloc[index].drop("PassengerId"))[0])

    print clf.score(train_features, train_labels)


def predictionWithMajorityVoting(train_features, train_labels, test_features):
    from sklearn.linear_model import LogisticRegression
    logistic = LogisticRegression()
    logistic.fit(train_features, train_labels)

    from sklearn.svm import SVC
    svc = SVC()
    svc.fit(train_features, train_labels)

    from sklearn.ensemble import RandomForestClassifier
    rfor = RandomForestClassifier(n_estimators=100)
    rfor.fit(train_features, train_labels)

    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(train_features, train_labels)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_features, train_lables)
    print "PassengerId,Survived"
    for index in range(len(test_features)):
        zeroCount = 0
        oneCount = 0
        lgprediction = int(logistic.predict(test_features.iloc[index].drop("PassengerId"))[0])
        if lgprediction==1:
            oneCount+=1
        else:
            zeroCount+=1

        svprediction = int(svc.predict(test_features.iloc[index].drop("PassengerId"))[0])
        if svprediction==1:
            oneCount+=1
        else:
            zeroCount+=1
        knprediction = int(knn.predict(test_features.iloc[index].drop("PassengerId"))[0])
        if knprediction==1:
            oneCount+=1
        else:
            zeroCount+=1
        rforprediction = int(rfor.predict(test_features.iloc[index].drop("PassengerId"))[0])
        if rforprediction==1:
            oneCount+=1
        else:
            zeroCount+=1
        gnprediction = int(gnb.predict(test_features.iloc[index].drop("PassengerId"))[0])
        if gnprediction==1:
            oneCount+=1
        else:
            zeroCount+=1
        if oneCount>zeroCount:
            prediction = 1
        else:
            prediction = 0
        print str(test_features.iloc[index]["PassengerId"].astype(int))+","+str(prediction)


if __name__=="__main__":
    print "main method"
    trainDF, testDF = loadData()
    trainDF, testDF = insertDummyForEmbark(trainDF, testDF)
    trainDF, testDF = preProcessingForFare(trainDF, testDF)
    trainDF, testDF = preProcessingForAge(trainDF, testDF)
    trainDF, testDF = preProcessingForCabin(trainDF, testDF)
    trainDF, testDF = preProcessingForFamily(trainDF, testDF)
    trainDF, testDF = preProcessingForGender(trainDF, testDF)
    trainDF, testDF = preProcessingForPclass(trainDF, testDF)

    #feed the training and testing data to different classifiers
    #lets drop columns that dont contribute
    #test_features = testDF.drop("PassengerId", axis = 1).copy()
    test_features = testDF.copy()

    #split the train data into features and labels
    train_lables = trainDF["Survived"]
    train_features = trainDF.drop("Survived", axis = 1)
    #print train_features.columns.values
    #print test_features.columns.values
    #predictWithLogisticRegression(train_features, train_lables, test_features)
    #predictionWithSVM(train_features, train_lables, test_features)
    #predictionWithGaussianNB(train_features, train_lables, test_features)
    #predictionWithKNN(train_features, train_lables, test_features)
    #predictionWithRandomForest(train_features, train_lables, test_features)
    #predictionWithMajorityVoting(train_features, train_lables, test_features)
