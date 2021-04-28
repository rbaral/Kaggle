'''
a factorization machine based model for loan prediction

Ref:
https://www.analyticsvidhya.com/blog/2018/01/factorization-machines/
https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/
'''
import pandas as pd
import xlearn as xl

import os

data_dir = "/Users/dur-rbaral-m/projects/test_projects/data/credit_prediction"

train = pd.read_csv(os.path.join(data_dir, "train.csv"))
import warnings
warnings.filterwarnings('ignore')


#For simplicity we will just take a few variables here:

cols = ['Education','ApplicantIncome','Loan_Status','Credit_History']
train_sub = train[cols]
train_sub['Credit_History'].fillna(0, inplace = True)
dict_ls = {'Y':1, 'N':0}
train_sub['Loan_Status'].replace(dict_ls, inplace = True)

#Next we will create a test set for testing the ffm model

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(train_sub, test_size = 0.3, random_state = 5)


def convert_to_ffm(df, type, numerics, categories, features):
    currentcode = len(numerics)
    catdict = {}
    catcodes = {}

    # Flagging categorical and numerical fields
    for x in numerics:
        catdict[x] = 0
    for x in categories:
        catdict[x] = 1

    nrows = df.shape[0]
    ncolumns = len(features)
    with open(os.path.join(data_dir,str(type) + "_ffm.txt"), "w") as text_file:

        # Looping over rows to convert each row to libffm format
        for n, r in enumerate(range(nrows)):
            datastring = ""
            datarow = df.iloc[r].to_dict()
            datastring += str(int(datarow['Loan_Status']))  # Set Target Variable here

            # For numerical fields, we are creating a dummy field here
            for i, x in enumerate(catdict.keys()):
                if (catdict[x] == 0):
                    datastring = datastring + " " + str(i) + ":" + str(i) + ":" + str(datarow[x])
                else:

                    # For a new field appearing in a training example
                    if (x not in catcodes):
                        catcodes[x] = {}
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode  # encoding the feature

                    # For already encoded fields
                    elif (datarow[x] not in catcodes[x]):
                        currentcode += 1
                        catcodes[x][datarow[x]] = currentcode  # encoding the feature

                    code = catcodes[x][datarow[x]]
                    datastring = datastring + " " + str(i) + ":" + str(int(code)) + ":1"

            datastring += '\n'
            text_file.write(datastring)


convert_to_ffm(X_train, "train", ["ApplicantIncome", "Credit_History"], ["Loan_Status", "Education"], ["ApplicantIncome", "Credit_History", "Education"])

convert_to_ffm(X_test, "test", ["ApplicantIncome", "Credit_History"], ["Loan_Status", "Education"], ["ApplicantIncome", "Credit_History", "Education"])


ffm_model = xl.create_ffm()

ffm_model.setTrain(os.path.join(data_dir, "train_ffm.txt"))

param = {'task':'binary',
         'lr':0.2,
         'lambda':0.002,
         'metric':'acc'}

# Start to train
# The trained model will be stored in model.out
ffm_model.fit(param, os.path.join(data_dir,'model.out'))

#The library also allows us to use cross-validation using the cv() function:

ffm_model.cv(param)
#Predictions can be done on the test set with the following code snippet:

# Prediction task
ffm_model.setTest(os.path.join(data_dir,"test_ffm.txt")) # Test data
ffm_model.setSigmoid() # Convert output to 0-1

# Start to predict
# The output result will be stored in output.txt
ffm_model.predict(os.path.join(data_dir,"model.out"), os.path.join(data_dir,"output.txt"))