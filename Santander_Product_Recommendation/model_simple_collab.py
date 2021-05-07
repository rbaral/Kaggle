"""
Ref:
https://www.kaggle.com/anokas/collaborative-filtering-btb-lb-0-01691
"""
import numpy as np
import os
import pandas as pd
import re
from collections import defaultdict, Counter
import zipfile
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb


# constants
DATA_DIR = "../../Kaggle_data/santander_product_recommendation"
WORK_DIR = "../output"
dataset = "santander-product-recommendation"

# Will unzip the files so that you can see them..
def unzip_files():
    train_file = os.path.join(DATA_DIR, "train_ver2.csv.zip")
    test_file = os.path.join(DATA_DIR, "test_ver2.csv.zip")
    sample_sub_file = os.path.join(DATA_DIR, "sample_submission.csv.zip")
    files = [train_file, test_file, sample_sub_file]
    for file in files:
        with zipfile.ZipFile(file, "r") as z:
            z.extractall(WORK_DIR)
    print("files in work dir are:")
    for item in os.listdir(WORK_DIR):
        print(item)

WORK_DIR = DATA_DIR
train_file = os.path.join(WORK_DIR, "train_ver2.csv")
test_file = os.path.join(WORK_DIR, "test_ver2.csv")
sample_sub_file = os.path.join(WORK_DIR, "sample_submission.csv")
result_file = os.path.join(WORK_DIR, "new_submission.csv")

# we only use following columns
#item cols and user id
item_cols = ['ncodpers', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
           'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
           'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
           'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
           'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
           'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
           'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
           'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

#user related features
user_features = ["age", "ind_nuevo", "sexo", "tiprel_1mes", "ind_actividad_cliente", "renta", "antiguedad", "segmento", "indrel_1mes", "ind_empleado", "nomprov"]#, "conyuemp"]#,

#feature weights
feature_weights = {"age":1.0,
           "ind_nuevo":0.5,
           "sexo":1.0,
           "tiprel_1mes":1.0,
           "ind_actividad_cliente":1.0,
           "renta":1.0,
           "antiguedad":0.5,
           "segmento":1.0,
           "indrel_1mes":0.5,
           "ind_empleado":1.0,
           "nomprov":0.25}


def label_encode(df, colname, weight=1.0):
    df_dummies = pd.get_dummies(df[colname], prefix=colname+"_")
    #add the feature weights
    for col in df_dummies.columns:
        df_dummies[col] = df_dummies[col]*weight
    df = pd.concat([df, df_dummies], axis=1)
    df.drop([colname], inplace=True, axis=1)
    return df


def get_data_training():
    """
    read training data
    """
    print("getting training data")

    usecols = item_cols + user_features
    print("using cols: ",usecols)
    df_train = pd.read_csv(train_file, usecols=usecols)
    #df_train = df_train.drop_duplicates()

    df_train = df_train.drop_duplicates(['ncodpers'], keep='last')

    #df_train_1 = df_train.drop_duplicates(['ncodpers'], keep='first')
    #df_train_2 = df_train.drop_duplicates(['ncodpers'], keep='last')
    #df_train = pd.concat([df_train_1, df_train_2])

    #ind_empleado Employee index: A active, B ex employed, F filial, N not employee, P pasive
    if "ind_empleado" in df_train.columns:
        #print("ind_empleado values ", Counter(df_train["ind_empleado"].values))
        df_train.fillna({"ind_empleado": "N"}, inplace=True)
        df_train = label_encode(df_train, "ind_empleado", weight=feature_weights["ind_empleado"])

    #for gender use V as default
    if "sexo" in df_train.columns:
        #print("gender values ", Counter(df_train["sexo"].values))
        df_train.fillna({"sexo":"V"}, inplace=True)
        df_train = label_encode(df_train, "sexo", weight=feature_weights["sexo"])

    #age
    if "age" in df_train.columns:
        #print("age values ", Counter(df_train["age"].values))
        # strip values
        df_train["age"] = df_train["age"].apply(lambda x: x.strip().replace("NA", "15") if type(x) == str else str(x))
        df_train.fillna({"age":"15"}, inplace=True)
        df_train["age"] = df_train["age"].astype("float")
        # now normalize the values
        min_max_scaler = preprocessing.MinMaxScaler()
        df_train[["age"]] = min_max_scaler.fit_transform(df_train[["age"]])
        df_train["age"] = df_train["age"]*feature_weights["age"]

    # for ind_nuevo use 1 as default; New customer Index. 1 if the customer registered in the last 6 months.
    if "ind_nuevo" in df_train.columns:
        #print("ind_nuevo values ", Counter(df_train["ind_nuevo"].values))
        df_train.fillna({"ind_nuevo": "1.0"}, inplace=True)
        df_train["ind_nuevo"] = df_train["ind_nuevo"].astype(float)
        df_train['ind_nuevo']= df_train['ind_nuevo']*[feature_weights["ind_nuevo"]]

    #for tiprel_1mes, use I as default; Customer relation type at the beginning of the month, A (active), I (inactive), P (former customer),R (Potential)
    if "tiprel_1mes" in df_train.columns:
        #print("tiprel_1mes values ", Counter(df_train["tiprel_1mes"].values))
        df_train.fillna({"tiprel_1mes":"I"}, inplace=True)
        df_train = label_encode(df_train, "tiprel_1mes", weight=feature_weights["tiprel_1mes"])

    #indrel_1mes Customer type at the beginning of the month ,1 (First/Primary customer), 2 (co-owner ),P (Potential),3 (former primary), 4(former co-owner)
    # for indrel_1mes, use 1.0 as default
    if "indrel_1mes" in df_train.columns:
        df_train.fillna({"indrel_1mes": "1.0"}, inplace=True)
        # there are duplicate type values, e.g., 2.0 as number and '2' as string
        for i in range(4):
            df_train["indrel_1mes"] = df_train["indrel_1mes"].apply(lambda x: str(float(i+1)) if (x==float(i+1) or x==(i+1) or x==str(i+1)) else x)
        #print("indrel_1mes values ", Counter(df_train["indrel_1mes"].values))
        df_train = label_encode(df_train, "indrel_1mes", weight=feature_weights["indrel_1mes"])

    #customers seniority in months
    if "antiguedad" in df_train.columns:
        #print("antiguedad values ", Counter(df_train["antiguedad"].values))
        df_train.fillna({"antiguedad": 1}, inplace=True)
        #strip values
        df_train["antiguedad"] = df_train["antiguedad"].apply(lambda x: x.strip().replace("NA", "1") if type(x)==str else str(x))
        df_train["antiguedad"] = df_train["antiguedad"].astype("float")
        #if negative use 1
        df_train["antiguedad"] = df_train["antiguedad"].apply(lambda x: 1.0 if x<0 else x)
        # now normalize the values
        min_max_scaler = preprocessing.MinMaxScaler()
        df_train[["antiguedad"]] = min_max_scaler.fit_transform(df_train[["antiguedad"]])
        df_train.loc[:, 'antiguedad'] *= feature_weights["antiguedad"]

    #gross income
    if "renta" in df_train.columns:
        #print("renta values ", Counter(df_train["renta"].values))
        df_train.fillna({"renta": 0}, inplace=True)
        #now normalize the values
        min_max_scaler = preprocessing.MinMaxScaler()
        df_train[["renta"]] = min_max_scaler.fit_transform(df_train[["renta"]])
        df_train.loc[:, 'renta'] *= feature_weights["renta"]

    # customer primary for the month or not
    if "indrel" in df_train.columns:
        # print("indrel values ", Counter(df_train["indrel"].values))
        df_train.fillna({"indrel": 0}, inplace=True)
        df_train["indrel"] = df_train["indrel"].astype(float)
        df_train.loc[:, 'indrel'] *= feature_weights["indrel"]

    # active or not
    if "ind_actividad_cliente" in df_train.columns:
        # print("ind_actividad_cliente values ", Counter(df_train["ind_actividad_cliente"].values))
        df_train.fillna({"ind_actividad_cliente": 0}, inplace=True)
        df_train["ind_actividad_cliente"] = df_train["ind_actividad_cliente"].astype(float)
        df_train.loc[:, 'ind_actividad_cliente'] *= feature_weights["ind_actividad_cliente"]

    #segmentation: 01 - VIP, 02 - Individuals 03 - college graduated
    if "segmento" in df_train.columns:
        print("segmento values ", Counter(df_train["segmento"].values))
        df_train.fillna({"segmento": "02"}, inplace=True)
        df_train = label_encode(df_train, "segmento", weight=feature_weights["segmento"])

    #province name
    if "nomprov" in df_train.columns:
        print("nomprov values ", Counter(df_train["nomprov"].values))
        df_train.fillna({"nomprov": "MADRID"}, inplace=True)
        df_train = label_encode(df_train, "nomprov", weight=feature_weights["nomprov"])

    #spouse index
    if "conyuemp" in df_train.columns:
        print("conyuemp values ", Counter(df_train["conyuemp"].values))
        df_train.fillna({"conyuemp": " "}, inplace=True)
        df_train = label_encode(df_train, "conyuemp", weight=feature_weights["conyuemp"])


    #for the rest fill missing values with 0
    df_train.fillna(0, inplace=True)
    return df_train


def predict_from_ensemble(x_train, y_train):
    model_logit = LogisticRegression(max_iter=1000)
    model_rf = RandomForestClassifier(n_estimators=200, max_depth=100, random_state=42)
    param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
    model_xgb = xgb.XGBClassifier(random_state=42, learning_rate=0.01, **param)
    #lgb model
    lgb_data = lgb.Dataset(x_train, label=y_train)
    lgb_params = {'learning_rate': 0.001}
    model_lgb = lgb.LGBMClassifier(**lgb_params)

    #first level models
    model_logit.fit(x_train, y_train)
    model_rf.fit(x_train, y_train)
    model_lgb.fit(x_train, y_train)

    preds_logit = model_logit.predict_proba(x_train)
    preds_rf = model_rf.predict_proba(x_train)
    preds_lgb = model_lgb.predict_proba(x_train)

    # use xgb in second layer
    ens_train = np.concatenate((preds_logit, preds_rf, preds_lgb), axis=1)
    model_xgb.fit(ens_train, y_train)
    preds = model_xgb.predict_proba(ens_train)
    return preds


def predict_from_lightgbm(x_train, y_train):
    lgb_params = {'learning_rate': 0.001}
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(x_train, y_train)
    preds = model_lgb.predict_proba(x_train)
    return preds


def predict_from_xgb(x_train, y_train):
    # specify parameters via map
    param = {'max_depth': 2, 'eta': 1, 'objective': 'binary:logistic'}
    model = xgb.XGBClassifier(random_state=42, learning_rate=0.01, **param)
    model.fit(x_train, y_train)
    # make prediction
    #data_dmatrix = xgb.DMatrix(data=x_train, label=y_train)
    preds = model.predict_proba(x_train)
    return preds


def predict_from_logit(x_train, y_train):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train, y_train)
    preds = clf.predict_proba(x_train)
    return preds


def predict_from_randomforest(x_train, y_train):
    clf = RandomForestClassifier(n_estimators=200, max_depth=100, random_state=42)
    clf.fit(x_train, y_train)
    preds = clf.predict_proba(x_train)
    return preds


def predict_feature_prob(ids):
    """
    for each feature, take it as a label and predict it using other features' values
    """
    print("predicting feature probability")
    #models = {}
    model_preds = {}
    id_preds = defaultdict(list)
    for feature_index, feature in enumerate(item_cols[1:]):
        #don't take the user id and user features as label to predict
        y_train = df_train[feature]
        # drop this feature along axis =1
        x_train = df_train.drop([feature, 'ncodpers'], axis=1)

        # train model for this feature as label
        #clf = LogisticRegression(max_iter=1000)
        #clf.fit(x_train, y_train)
        #model_predict_prob = clf.predict_proba(x_train)
        p_train = predict_from_ensemble(x_train, y_train)[:, 1]#model_predict_prob[:, 1]
        # accumulate the model and its prediction prob for each feature
        #models[feature] = clf
        model_preds[feature] = p_train
        # for every user, add the prediction on each feature
        for id, p in zip(ids, p_train):
            #if the feature already defined for this user, the average it
            if id in id_preds and len(id_preds[id])>=feature_index+1:
                new_score = np.average([id_preds[id][feature_index], p])
                id_preds[id][feature_index] = new_score
            else:
                id_preds[id].append(p)
        print("label ", feature, ", roc is:", roc_auc_score(y_train, p_train))
    return id_preds


def get_used_features():
    """
    for each user, get the already used features
    """
    print("filtering used features")
    # is this user an active user? if any of the features is non-zero take this user as an active user
    used_features = {}
    for row in df_train.values:
        row = list(row)
        id = row.pop(0)
        active = [c[0] for c in zip(df_train.columns[1:], row) if (c[0] in item_cols and c[1] > 0)]
        if id not in used_features:
            used_features[id] = []
        used_features[id].extend(active)
    return used_features


def predict_user_items(id_preds, used_features, df_sample):
    print("generating sample output file")
    train_preds = {}
    for id, p in id_preds.items():
        new_items = [i for i in zip(item_cols[1:], p) if i[0] not in used_features[id]]
        sorted_preds = sorted(new_items, key=lambda i: i[1], reverse=True)[:7]
        preds = [i[0] for i in sorted_preds]
        train_preds[id] = preds

    test_preds = []
    for row in df_sample.values:
        id = row[0]
        p = train_preds[id]
        test_preds.append(' '.join(p))

    df_sample['added_products'] = test_preds
    df_sample.to_csv(result_file, index=False)


df_train = get_data_training()

#rename column headers to prevent lgbm breaking
df_train = df_train.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

for col in df_train.columns:
    print(col,"...",df_train[col].dtype)

df_sample = pd.read_csv(sample_sub_file)
ids = df_train['ncodpers'].values
ids_unique = list(df_train["ncodpers"].unique())
print("total users ", len(ids), " total unique users ", len(ids_unique))

# for every feature, attempt to predict it using other features
id_preds = predict_feature_prob(ids)

used_features = get_used_features()

predict_user_items(id_preds, used_features, df_sample)