'''
ref: https://github.com/kalpishs/House-Prices-Advanced-Regression-Techniques/blob/master/Final%20Submission/data_exploration.py
setup xgboost ref: https://www.ibm.com/developerworks/community/blogs/jfp/entry/Installing_XGBoost_For_Anaconda_on_Windows?lang=en

TODOs: check this also
https://www.kaggle.com/amitchoudhary/house-prices-advanced-regression-techniques/script-v6/run/512428
'''

from HeaderFiles import*

trainData = pd.read_csv("data/train.csv")
testData = pd.read_csv("data/test.csv")

allData = pd.concat(
    (trainData.loc[:,'MSSubClass':'SaleCondition'],testData.loc[:,'MSSubClass':'SaleCondition']), ignore_index = True)

warnings.simplefilter('ignore', np.RankWarning)
x = allData.loc[np.logical_not(allData["LotFrontage"].isnull()), "LotArea"]
y = allData.loc[np.logical_not(allData["LotFrontage"].isnull()), "LotFrontage"]

t = (x<=25000) & (y<=150) #Extract values
p = np.polyfit(x[t], y[t], 1)

allData.loc[allData['LotFrontage'].isnull(), 'LotFrontage'] = np.polyval(p, allData.loc[allData['LotFrontage'].isnull(), 'LotArea'])


#handle missing values
allData = allData.fillna({
    'Alley': 'NoAlley',
    'MasVnrType': 'None',
    'FireplaceQu': 'NoFireplace',
    'GarageType': 'NoGarage',
    'GarageFinish': 'NoGarage',
    'GarageQual': 'NoGarage',
    'GarageCond': 'NoGarage',
    'BsmtFullBath': 0,
    'BsmtHalfBath': 0,
    'BsmtQual': 'NoBsmt',
    'BsmtCond': 'NoBsmt',
    'BsmtExposure': 'NoBsmt',
    'BsmtFinType1': 'NoBsmt',
    'BsmtFinType2': 'NoBsmt',
    'KitchenQual': 'TA',
    'MSZoning': 'RL',
    'Utilities': 'AllPub',
    'Exterior1st': 'VinylSd',
    'Exterior2nd': 'VinylSd',
    'Functional': 'Typ',
    'PoolQC': 'NoPool',
    'Fence': 'NoFence',
    'MiscFeature': 'None',
    'Electrical': 'SBrkr'

})

allData.loc[allData.SaleCondition.isnull(), 'SaleCondition'] = 'Normal'
allData.loc[allData.SaleCondition.isnull(), 'SaleType'] = 'WD'
allData.loc[allData.MasVnrType == 'None', 'MasVnrArea'] = 0
allData.loc[allData.BsmtFinType1=='NoBsmt', 'BsmtFinSF1'] = 0
allData.loc[allData.BsmtFinType2=='NoBsmt', 'BsmtFinSF2'] = 0
allData.loc[allData.BsmtFinSF1.isnull(), 'BsmtFinSF1'] = allData.BsmtFinSF1.median()
allData.loc[allData.BsmtQual=='NoBsmt', 'BsmtUnfSF'] = 0
allData.loc[allData.BsmtUnfSF.isnull(), 'BsmtUnfSF'] = allData.BsmtUnfSF.median()
allData.loc[allData.BsmtQual=='NoBsmt', 'TotalBsmtSF'] = 0

# only one is null and it has type Detchd
allData.loc[allData['GarageArea'].isnull(), 'GarageArea'] = allData.loc[allData['GarageType']=='Detchd', 'GarageArea'].mean()
allData.loc[allData['GarageCars'].isnull(), 'GarageCars'] = allData.loc[allData['GarageType']=='Detchd', 'GarageCars'].median()

# where we have order we will use numeric
allData = allData.replace({'Utilities': {'AllPub': 1, 'NoSeWa': 0, 'NoSewr': 0, 'ELO': 0},
                             'Street': {'Pave': 1, 'Grvl': 0 },
                             'FireplaceQu': {'Ex': 5,
                                            'Gd': 4,
                                            'TA': 3,
                                            'Fa': 2,
                                            'Po': 1,
                                            'NoFireplace': 0
                                            },
                             'Fence': {'GdPrv': 2,
                                       'GdWo': 2,
                                       'MnPrv': 1,
                                       'MnWw': 1,
                                       'NoFence': 0},
                             'ExterQual': {'Ex': 5,
                                            'Gd': 4,
                                            'TA': 3,
                                            'Fa': 2,
                                            'Po': 1
                                            },
                             'ExterCond': {'Ex': 5,
                                            'Gd': 4,
                                            'TA': 3,
                                            'Fa': 2,
                                            'Po': 1
                                            },
                             'BsmtQual': {'Ex': 5,
                                            'Gd': 4,
                                            'TA': 3,
                                            'Fa': 2,
                                            'Po': 1,
                                            'NoBsmt': 0},
                             'BsmtExposure': {'Gd': 3,
                                            'Av': 2,
                                            'Mn': 1,
                                            'No': 0,
                                            'NoBsmt': 0},
                             'BsmtCond': {'Ex': 5,
                                            'Gd': 4,
                                            'TA': 3,
                                            'Fa': 2,
                                            'Po': 1,
                                            'NoBsmt': 0},
                             'GarageQual': {'Ex': 5,
                                            'Gd': 4,
                                            'TA': 3,
                                            'Fa': 2,
                                            'Po': 1,
                                            'NoGarage': 0},
                             'GarageCond': {'Ex': 5,
                                            'Gd': 4,
                                            'TA': 3,
                                            'Fa': 2,
                                            'Po': 1,
                                            'NoGarage': 0},
                             'KitchenQual': {'Ex': 5,
                                            'Gd': 4,
                                            'TA': 3,
                                            'Fa': 2,
                                            'Po': 1},
                             'Functional': {'Typ': 0,
                                            'Min1': 1,
                                            'Min2': 1,
                                            'Mod': 2,
                                            'Maj1': 3,
                                            'Maj2': 4,
                                            'Sev': 5,
                                            'Sal': 6}
                            })

allData = allData.replace({'CentralAir': {'Y': 1,
                                            'N': 0}})
allData = allData.replace({'PavedDrive': {'Y': 1,
                                            'P': 0,
                                            'N': 0}})

print allData