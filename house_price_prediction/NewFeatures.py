from DataAnalysis import *

newer_dwelling = allData.MSSubClass.replace({20: 1,
                                            30: 0,
                                            40: 0,
                                            45: 0,
                                            50: 0,
                                            60: 1,
                                            70: 0,
                                            75: 0,
                                            80: 0,
                                            85: 0,
                                            90: 0,
                                           120: 1,
                                           150: 0,
                                           160: 0,
                                           180: 0,
                                           190: 0})
newer_dwelling.name = 'newer_dwelling'

allData = allData.replace({'MSSubClass': {20: 'SubClass_20',
                                            30: 'SubClass_30',
                                            40: 'SubClass_40',
                                            45: 'SubClass_45',
                                            50: 'SubClass_50',
                                            60: 'SubClass_60',
                                            70: 'SubClass_70',
                                            75: 'SubClass_75',
                                            80: 'SubClass_80',
                                            85: 'SubClass_85',
                                            90: 'SubClass_90',
                                           120: 'SubClass_120',
                                           150: 'SubClass_150',
                                           160: 'SubClass_160',
                                           180: 'SubClass_180',
                                           190: 'SubClass_190'}})

# The idea is good quality should rise price, poor quality - reduce price
overall_poor_qu = allData.OverallQual.copy()
overall_poor_qu = 5 - overall_poor_qu
overall_poor_qu[overall_poor_qu < 0] = 0
overall_poor_qu.name = 'overall_poor_qu'

overall_good_qu = allData.OverallQual.copy()
overall_good_qu = overall_good_qu - 5
overall_good_qu[overall_good_qu < 0] = 0
overall_good_qu.name = 'overall_good_qu'

overall_poor_cond = allData.OverallCond.copy()
overall_poor_cond = 5 - overall_poor_cond
overall_poor_cond[overall_poor_cond < 0] = 0
overall_poor_cond.name = 'overall_poor_cond'

overall_good_cond = allData.OverallCond.copy()
overall_good_cond = overall_good_cond - 5
overall_good_cond[overall_good_cond < 0] = 0
overall_good_cond.name = 'overall_good_cond'

exter_poor_qu = allData.ExterQual.copy()
exter_poor_qu[exter_poor_qu < 3] = 1
exter_poor_qu[exter_poor_qu >= 3] = 0
exter_poor_qu.name = 'exter_poor_qu'

exter_good_qu = allData.ExterQual.copy()
exter_good_qu[exter_good_qu <= 3] = 0
exter_good_qu[exter_good_qu > 3] = 1
exter_good_qu.name = 'exter_good_qu'

exter_poor_cond = allData.ExterCond.copy()
exter_poor_cond[exter_poor_cond < 3] = 1
exter_poor_cond[exter_poor_cond >= 3] = 0
exter_poor_cond.name = 'exter_poor_cond'

exter_good_cond = allData.ExterCond.copy()
exter_good_cond[exter_good_cond <= 3] = 0
exter_good_cond[exter_good_cond > 3] = 1
exter_good_cond.name = 'exter_good_cond'

bsmt_poor_cond = allData.BsmtCond.copy()
bsmt_poor_cond[bsmt_poor_cond < 3] = 1
bsmt_poor_cond[bsmt_poor_cond >= 3] = 0
bsmt_poor_cond.name = 'bsmt_poor_cond'

bsmt_good_cond = allData.BsmtCond.copy()
bsmt_good_cond[bsmt_good_cond <= 3] = 0
bsmt_good_cond[bsmt_good_cond > 3] = 1
bsmt_good_cond.name = 'bsmt_good_cond'

garage_poor_qu = allData.GarageQual.copy()
garage_poor_qu[garage_poor_qu < 3] = 1
garage_poor_qu[garage_poor_qu >= 3] = 0
garage_poor_qu.name = 'garage_poor_qu'

garage_good_qu = allData.GarageQual.copy()
garage_good_qu[garage_good_qu <= 3] = 0
garage_good_qu[garage_good_qu > 3] = 1
garage_good_qu.name = 'garage_good_qu'

garage_poor_cond = allData.GarageCond.copy()
garage_poor_cond[garage_poor_cond < 3] = 1
garage_poor_cond[garage_poor_cond >= 3] = 0
garage_poor_cond.name = 'garage_poor_cond'

garage_good_cond = allData.GarageCond.copy()
garage_good_cond[garage_good_cond <= 3] = 0
garage_good_cond[garage_good_cond > 3] = 1
garage_good_cond.name = 'garage_good_cond'

kitchen_poor_qu = allData.KitchenQual.copy()
kitchen_poor_qu[kitchen_poor_qu < 3] = 1
kitchen_poor_qu[kitchen_poor_qu >= 3] = 0
kitchen_poor_qu.name = 'kitchen_poor_qu'

kitchen_good_qu = allData.KitchenQual.copy()
kitchen_good_qu[kitchen_good_qu <= 3] = 0
kitchen_good_qu[kitchen_good_qu > 3] = 1
kitchen_good_qu.name = 'kitchen_good_qu'

qu_list = pd.concat((overall_poor_qu, overall_good_qu, overall_poor_cond, overall_good_cond, exter_poor_qu,
                     exter_good_qu, exter_poor_cond, exter_good_cond, bsmt_poor_cond, bsmt_good_cond, garage_poor_qu,
                     garage_good_qu, garage_poor_cond, garage_good_cond, kitchen_poor_qu, kitchen_good_qu), axis=1)

bad_heating = allData.HeatingQC.replace({'Ex': 0,
                                          'Gd': 0,
                                          'TA': 0,
                                          'Fa': 1,
                                          'Po': 1})
bad_heating.name = 'bad_heating'

MasVnrType_Any = allData.MasVnrType.replace({'BrkCmn': 1,
                                              'BrkFace': 1,
                                              'CBlock': 1,
                                              'Stone': 1,
                                              'None': 0})
MasVnrType_Any.name = 'MasVnrType_Any'

SaleCondition_PriceDown = allData.SaleCondition.replace({'Abnorml': 1,
                                                          'Alloca': 1,
                                                          'AdjLand': 1,
                                                          'Family': 1,
                                                          'Normal': 0,
                                                          'Partial': 0})
SaleCondition_PriceDown.name = 'SaleCondition_PriceDown'

Neighborhood_Good = pd.DataFrame(np.zeros((allData.shape[0], 1)), columns=['Neighborhood_Good'])
Neighborhood_Good[allData.Neighborhood == 'NridgHt'] = 1
Neighborhood_Good[allData.Neighborhood == 'Crawfor'] = 1
Neighborhood_Good[allData.Neighborhood == 'StoneBr'] = 1
Neighborhood_Good[allData.Neighborhood == 'Somerst'] = 1
Neighborhood_Good[allData.Neighborhood == 'NoRidge'] = 1

svm = SVC(C=100, gamma=0.0001, kernel='rbf')
pc = pd.Series(np.zeros(trainData.shape[0]))

pc[:] = 'pc1'
pc[trainData.SalePrice >= 150000] = 'pc2'
pc[trainData.SalePrice >= 220000] = 'pc3'
columns_for_pc = ['Exterior1st', 'Exterior2nd', 'RoofMatl', 'Condition1', 'Condition2', 'BldgType']
X_t = pd.get_dummies(trainData.loc[:, columns_for_pc], sparse=True)
svm.fit(X_t, pc)  # Training
pc_pred = svm.predict(X_t)

p = trainData.SalePrice / 100000

price_category = pd.DataFrame(np.zeros((allData.shape[0], 1)), columns=['pc'])
X_t = pd.get_dummies(allData.loc[:, columns_for_pc], sparse=True)
pc_pred = svm.predict(X_t)
price_category[pc_pred == 'pc2'] = 1
price_category[pc_pred == 'pc3'] = 2

price_category = price_category.to_sparse()

season = allData.MoSold.replace({1: 0,
                                  2: 0,
                                  3: 0,
                                  4: 1,
                                  5: 1,
                                  6: 1,
                                  7: 1,
                                  8: 0,
                                  9: 0,
                                  10: 0,
                                  11: 0,
                                  12: 0})
season.name = 'season'

all_data = allData.replace({'MoSold': {1: 'Yan',
                                        2: 'Feb',
                                        3: 'Mar',
                                        4: 'Apr',
                                        5: 'May',
                                        6: 'Jun',
                                        7: 'Jul',
                                        8: 'Avg',
                                        9: 'Sep',
                                        10: 'Oct',
                                        11: 'Nov',
                                        12: 'Dec'}})

reconstruct = pd.DataFrame(np.zeros((allData.shape[0], 1)), columns=['Reconstruct'])
reconstruct[allData.YrSold < allData.YearRemodAdd] = 1
reconstruct = reconstruct.to_sparse()

recon_after_buy = pd.DataFrame(np.zeros((allData.shape[0], 1)), columns=['ReconstructAfterBuy'])
recon_after_buy[allData.YearRemodAdd >= allData.YrSold] = 1
recon_after_buy = recon_after_buy.to_sparse()

build_eq_buy = pd.DataFrame(np.zeros((allData.shape[0], 1)), columns=['Build.eq.Buy'])
build_eq_buy[allData.YearBuilt >= allData.YrSold] = 1
build_eq_buy = build_eq_buy.to_sparse()

allData.YrSold = 2010 - allData.YrSold
year_map = pd.concat(
    pd.Series('YearGroup' + str(i + 1), index=range(1871 + i * 20, 1891 + i * 20)) for i in range(0, 7))
allData.GarageYrBlt = allData.GarageYrBlt.map(year_map)
allData.loc[all_data['GarageYrBlt'].isnull(), 'GarageYrBlt'] = 'NoGarage'

allData.YearBuilt = allData.YearBuilt.map(year_map)
allData.YearRemodAdd = allData.YearRemodAdd.map(year_map)