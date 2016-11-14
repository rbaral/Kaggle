__author__ = 'rbaral'
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import GridSearchCV, StratifiedKFold


def testNullCount():
    df = pd.DataFrame({'a':[1,2,np.nan], 'b':[np.nan,1,np.nan]})
    print df["b"].isnull().sum(), df["b"].count(),len(df["b"])-df["b"].count()

def testGraph():
    df =pd.DataFrame({'name':['a','b','c','d'],'age':[20,20,30,40],'hour':[5, 5, 7, 5]})
    sns.set_style('whitegrid')
    sns.factorplot('name','age', data = df)

    sns.plt.show()

if __name__=="__main__":
    print "main method"
    #testGraph()
    print np.linspace(1000, 15000, 100)
    #testNullCount()