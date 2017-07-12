'''
data analysis of the zillow zestimate

Ref:
https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-zillow-prize
'''

import os, sys

dataDir = "H:\scis\Data_Mining\data\Kaggle\zillow_home" #"U:\scis\DataMining\data\Kaggle\zillow_home"
train_file_16 = os.path.join(dataDir, "train_2016.csv")
prop_file_16 = os.path.join(dataDir, "properties_2016.csv")
data_dict_file = os.path.join(dataDir, "zillow_data_dictionary.xlsx")

from ggplot import *
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

def configurePandas():
    pd.options.mode.chained_assignment = None
    pd.options.display.max_columns = 999


def analyzeTrainData():
    train_df = pd.read_csv(train_file_16, parse_dates=["transactiondate"])
    # print(train_df.shape)
    # print(train_df.head())
    # let us analyze the target field "logerror"
    # make simple scatter plot to see the distribution of logerror across the entries
    '''
    plt.figure(figsize=(8, 6))
    plt.scatter(range(train_df.shape[0]), np.sort(train_df.logerror.values))
    plt.xlabel('index', fontsize=12)
    plt.ylabel('logerror', fontsize=12)
    plt.show()
    train_df["logerror"].hist(bins = 50)
    plt.show()
    '''
    # lets remove the outliers
    ulimit = np.percentile(train_df.logerror.values, 99)
    print(ulimit)
    llimit = np.percentile(train_df.logerror.values, 1)
    train_df['logerror'].ix[train_df['logerror'] > ulimit] = ulimit
    train_df['logerror'].ix[train_df['logerror'] < llimit] = llimit

    # plot the pattern of logerror
    '''
    plt.figure(figsize=(12, 8))
    sns.distplot(train_df.logerror.values, bins=50, kde=False)
    plt.xlabel('logerror', fontsize=12)
    plt.show()
    #can be plotted using pandas dataframe as well
    #train_df["logerror"].hist(bins=50)
    #plt.show()
    #we can see that the logerror now has a normal distribution
    '''

    # let us explore the transactions made on each month
    # initialize the transaction month
    train_df['transaction_month'] = train_df['transactiondate'].dt.month
    month_values = train_df['transaction_month'].value_counts()
    # plot the monthly distribution of the records
    '''
    plt.figure(figsize=(12, 6))
    sns.barplot(month_values.index, month_values.values, alpha=0.8, color=color[3])
    plt.xticks(rotation='vertical')
    plt.xlabel('Month of transaction', fontsize=12)
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.show()
    #the same can be done to get the number values as
    #print(train_df['transaction_month'].value_counts())
    '''

    # lets see the parcelid field distribution
    # print((train_df['parcelid'].value_counts().reset_index())['parcelid'].value_counts())
    # we can see that most of the entries occur just once, few occur twice, and only one occurs thrice

'''
plots the bar chart of fields with missing values
'''
def plotFieldsWithMissingValues(df):
    missing_df = df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df = missing_df.ix[missing_df['missing_count'] > 0]
    missing_df = missing_df.sort_values(by='missing_count')

    ind = np.arange(missing_df.shape[0])
    width = 0.9
    fig, ax = plt.subplots(figsize=(12, 18))
    # every column/field is represented by a rectangle whose length represents the count of missing values
    rects = ax.barh(ind, missing_df.missing_count.values, color='red')
    # the y-axis of the plot is based on the index of the fields
    ax.set_yticks(ind)
    ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
    ax.set_xlabel("Count of missing values")
    ax.set_title("Number of missing values in each column")
    plt.show()


def analyzePropertiesFile():
    # exploring the properties file
    prop_df = pd.read_csv(prop_file_16)

    # let us see the missing values for every field
    #print(prop_df.apply(lambda x: sum(x.isnull()), axis=0))
    # it looks like most of the fields have missing values
    #lets plot the fields and the count of missing values in each field

    #plotFieldsWithMissingValues(prop_df)


    #let us see the trend of latitude and longitude values using a joint plot
    plt.figure(figsize=(12, 12))
    sns.jointplot(x=prop_df.latitude.values, y=prop_df.longitude.values, size=10)
    plt.ylabel('Longitude', fontsize=12)
    plt.xlabel('Latitude', fontsize=12)
    plt.show()


def analyzeTrainAndPropFile():
    #merging train and prop file and analyzing the trend
    train_df = pd.read_csv(train_file_16, parse_dates=["transactiondate"])
    #lets remove the outliers
    ulimit = np.percentile(train_df.logerror.values, 99)
    print(ulimit)
    llimit = np.percentile(train_df.logerror.values, 1)
    train_df['logerror'].ix[train_df['logerror'] > ulimit] = ulimit
    train_df['logerror'].ix[train_df['logerror'] < llimit] = llimit

    prop_df = pd.read_csv(prop_file_16)
    #left join of two dataframes
    train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')

    # configure pandas to display only 65 rows at max
    pd.options.display.max_rows = 65
    #get the data type of each column
    dtype_df = train_df.dtypes.reset_index()
    dtype_df.columns = ["Count", "Column Type"]
    #print(dtype_df)
    #can group the fields based on the data type
    #print(dtype_df.groupby("Column Type").aggregate('count').reset_index())

    #find the proportion of times a column has values missing
    missing_df = train_df.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['column_name', 'missing_count']
    missing_df['missing_ratio'] = missing_df['missing_count'] / train_df.shape[0]
    #get the columns with missing_ratio > 0.999
    #missing_df.ix[missing_df['missing_ratio'] > 0.999]

'''
performs analysis of impact of the float variables
on the training+prop dataframe
'''
def performUnivariateAnalysis():
    # merging train and prop file and analyzing the trend
    train_df = pd.read_csv(train_file_16, parse_dates=["transactiondate"])
    # lets remove the outliers
    ulimit = np.percentile(train_df.logerror.values, 99)

    llimit = np.percentile(train_df.logerror.values, 1)
    train_df['logerror'].ix[train_df['logerror'] > ulimit] = ulimit
    train_df['logerror'].ix[train_df['logerror'] < llimit] = llimit

    prop_df = pd.read_csv(prop_file_16)
    # left join of two dataframes
    train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')
    # use mean values to impute missing values and compute correlation coefficients #
    mean_values = train_df.mean(axis=0)
    train_df_new = train_df.fillna(mean_values, inplace=True)

    # Now let us look at the correlation coefficient of each of these variables #
    x_cols = [col for col in train_df_new.columns if col not in ['logerror'] if train_df_new[col].dtype == 'float64']

    labels = []
    values = []
    for col in x_cols:
        labels.append(col)
        values.append(np.corrcoef(train_df_new[col].values, train_df_new.logerror.values)[0, 1])
    corr_df = pd.DataFrame({'col_labels': labels, 'corr_values': values})
    corr_df = corr_df.sort_values(by='corr_values')

    #plot the correlation
    '''
    ind = np.arange(len(labels))
    width = 0.9
    fig, ax = plt.subplots(figsize=(12, 40))
    rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
    ax.set_yticks(ind)
    ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
    ax.set_xlabel("Correlation coefficient")
    ax.set_title("Correlation coefficient of the variables")
    # autolabel(rects)
    plt.show()
    '''

    #the correlation seems to be quite low
    #there are some variables with no correlation, let us find occurence of these variables
    corr_zero_cols = ['assessmentyear', 'storytypeid', 'pooltypeid2', 'pooltypeid7', 'pooltypeid10', 'poolcnt',
                      'decktypeid', 'buildingclasstypeid']
    #for col in corr_zero_cols:
    #    print(col, len(train_df_new[col].unique()))
    #let us see the variables with some correlation
    corr_df_sel = corr_df.ix[(corr_df['corr_values'] > 0.02) | (corr_df['corr_values'] < -0.01)]
    #print(corr_df_sel)

    #plot the heatmap of the correlation of these variables
    '''
    cols_to_use = corr_df_sel.col_labels.tolist()

    temp_df = train_df[cols_to_use]
    corrmat = temp_df.corr(method='spearman')
    f, ax = plt.subplots(figsize=(8, 8))

    # Draw the heatmap using seaborn
    sns.heatmap(corrmat, vmax=1., square=True)
    plt.title("Important variables correlation map", fontsize=15)
    plt.show()
    '''

    #analyze the correlation of field "finishedsquarefeet12" and "logerror"
    '''
    col = "finishedsquarefeet12"
    ulimit = np.percentile(train_df[col].values, 99.5)
    llimit = np.percentile(train_df[col].values, 0.5)
    train_df[col].ix[train_df[col] > ulimit] = ulimit
    train_df[col].ix[train_df[col] < llimit] = llimit

    plt.figure(figsize=(12, 12))
    sns.jointplot(x=train_df.finishedsquarefeet12.values, y=train_df.logerror.values, size=10, color=color[4])
    plt.ylabel('Log Error', fontsize=12)
    plt.xlabel('Finished Square Feet 12', fontsize=12)
    plt.title("Finished square feet 12 Vs Log error", fontsize=15)
    plt.show()
    '''

    #analyze the correlation of field "calculatedfinishedsquarefeet" and "logerror"
    '''
    col = "calculatedfinishedsquarefeet"
    ulimit = np.percentile(train_df[col].values, 99.5)
    llimit = np.percentile(train_df[col].values, 0.5)
    train_df[col].ix[train_df[col] > ulimit] = ulimit
    train_df[col].ix[train_df[col] < llimit] = llimit

    plt.figure(figsize=(12, 12))
    sns.jointplot(x=train_df.calculatedfinishedsquarefeet.values, y=train_df.logerror.values, size=10, color=color[5])
    plt.ylabel('Log Error', fontsize=12)
    plt.xlabel('Calculated finished square feet', fontsize=12)
    plt.title("Calculated finished square feet Vs Log error", fontsize=15)
    plt.show()
    '''

    #plot the distribution of bathroomcnt field
    '''
    plt.figure(figsize=(12, 8))
    sns.countplot(x="bathroomcnt", data=train_df)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Bathroom', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.title("Frequency of Bathroom count", fontsize=15)
    plt.show()
    #what is the 2.279 value, is it mean?
    print(train_df['bathroomcnt'].mean())
    '''

    #let us see how the value of "bathroomcnt" impacts "logerror"
    '''
    plt.figure(figsize=(12, 8))
    sns.boxplot(x="bathroomcnt", y="logerror", data=train_df)
    plt.ylabel('Log error', fontsize=12)
    plt.xlabel('Bathroom Count', fontsize=12)
    plt.xticks(rotation='vertical')
    plt.title("How log error changes with bathroom count?", fontsize=15)
    plt.show()
    '''


    #let us see the impact of bedroomcnt to logerror
    '''
    train_df['bedroomcnt'].ix[train_df['bedroomcnt'] > 7] = 7
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='bedroomcnt', y='logerror', data=train_df)
    plt.xlabel('Bedroom count', fontsize=12)
    plt.ylabel('Log Error', fontsize=12)
    plt.show()
    '''

    #let us see the impact of taxamount to logerror
    '''
    col = "taxamount"
    ulimit = np.percentile(train_df[col].values, 99.5)
    llimit = np.percentile(train_df[col].values, 0.5)
    train_df[col].ix[train_df[col] > ulimit] = ulimit
    train_df[col].ix[train_df[col] < llimit] = llimit

    plt.figure(figsize=(12, 12))
    sns.jointplot(x=train_df['taxamount'].values, y=train_df['logerror'].values, size=10, color='g')
    plt.ylabel('Log Error', fontsize=12)
    plt.xlabel('Tax Amount', fontsize=12)
    plt.title("Tax Amount Vs Log error", fontsize=15)
    plt.show()
    '''

    #yearbuilt vs logerror
    '''
    g = ggplot(aes(x='yearbuilt', y='logerror'), data=train_df) + \
    geom_point(color='steelblue', size=1) + \
    stat_smooth()
    g.show()
    '''

    #lat lon vs logerror
    '''
    g =ggplot(aes(x='latitude', y='longitude', color='logerror'), data=train_df) + \
    geom_point() + \
    scale_color_gradient(low='red', high='blue')
    g.show()
    '''

    #finishedsquarefoot taxamount vs logerror
    '''
    g = ggplot(aes(x='finishedsquarefeet12', y='taxamount', color='logerror'), data=train_df) + \
    geom_point(alpha=0.7) + \
    scale_color_gradient(low='pink', high='blue')
    g.show()
    '''

    #a bird art plot
    '''
    g = ggplot(aes(x='finishedsquarefeet12', y='taxamount', color='logerror'), data=train_df) + \
    geom_now_its_art()
    '''

    #now analyze the feature importances using RegressorTree

    train_y = train_df['logerror'].values
    cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]
    train_df['transaction_month'] = train_df['transactiondate'].dt.month
    train_df = train_df.drop(['parcelid', 'logerror', 'transactiondate', 'transaction_month']+cat_cols, axis=1)
    feat_names = train_df.columns.values

    from sklearn import ensemble
    model = ensemble.ExtraTreesRegressor(n_estimators=25, max_depth=30, max_features=0.3, n_jobs=-1, random_state=0)
    model.fit(train_df, train_y)

    ## plot the importances ##
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
    indices = np.argsort(importances)[::-1][:20]

    plt.figure(figsize=(12,12))
    plt.title("Feature importances")
    plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
    plt.xlim([-1, len(indices)])
    plt.show()





def dataAnalysis():
    #analyzeTrainData()
    #analyzePropertiesFile()
    #analyzeTrainAndPropFile()
    performUnivariateAnalysis()

if __name__=="__main__":
    print("main started")
    dataAnalysis()