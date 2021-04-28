"""
Ref:
https://www.kaggle.com/anokas/collaborative-filtering-btb-lb-0-01691
"""
import numpy as np
import os
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from collections import defaultdict, Counter
import zipfile

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
user_features = ["age", "ind_nuevo", "sexo", "tiprel_1mes", "ind_actividad_cliente", "renta", "antiguedad", "segmento"]#"indrel_1mes", "ind_empleado", "nomprov", "conyuemp"]#,

def label_encode(df, colname):
    df_dummies = pd.get_dummies(df[colname], prefix=colname+"_")
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
    #df_train_1 = df_train.drop_duplicates(['ncodpers'], keep='first')
    df_train = df_train.drop_duplicates(['ncodpers'], keep='last')
    #df_train = pd.concat([df_train_1, df_train_2])
    #df_train = df_train.drop_duplicates(['ncodpers'], keep='last')

    #df_train = df_train.drop_duplicates()

    #ind_empleado Employee index: A active, B ex employed, F filial, N not employee, P pasive
    if "ind_empleado" in df_train.columns:
        #print("ind_empleado values ", Counter(df_train["ind_empleado"].values))
        df_train.fillna({"ind_empleado": "N"}, inplace=True)
        df_train = label_encode(df_train, "ind_empleado")

    #for gender use V as default
    if "sexo" in df_train.columns:
        #print("gender values ", Counter(df_train["sexo"].values))
        df_train.fillna({"sexo":"V"}, inplace=True)
        df_train = label_encode(df_train, "sexo")

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

    # for ind_nuevo use 1 as default; New customer Index. 1 if the customer registered in the last 6 months.
    if "ind_nuevo" in df_train.columns:
        #print("ind_nuevo values ", Counter(df_train["ind_nuevo"].values))
        df_train.fillna({"ind_nuevo": "1"}, inplace=True)

    #for tiprel_1mes, use I as default; Customer relation type at the beginning of the month, A (active), I (inactive), P (former customer),R (Potential)
    if "tiprel_1mes" in df_train.columns:
        #print("tiprel_1mes values ", Counter(df_train["tiprel_1mes"].values))
        df_train.fillna({"tiprel_1mes":"I"}, inplace=True)
        df_train = label_encode(df_train, "tiprel_1mes")

    #indrel_1mes Customer type at the beginning of the month ,1 (First/Primary customer), 2 (co-owner ),P (Potential),3 (former primary), 4(former co-owner)
    # for indrel_1mes, use I as default
    if "indrel_1mes" in df_train.columns:
        #print("indrel_1mes values ", Counter(df_train["indrel_1mes"].values))
        df_train.fillna({"indrel_1mes": "I"}, inplace=True)
        df_train = label_encode(df_train, "indrel_1mes")

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

    #gross income
    if "renta" in df_train.columns:
        #print("renta values ", Counter(df_train["renta"].values))
        df_train.fillna({"renta": 0}, inplace=True)
        #now normalize the values
        min_max_scaler = preprocessing.MinMaxScaler()
        df_train[["renta"]] = min_max_scaler.fit_transform(df_train[["renta"]])

    # customer primary for the month or not
    if "indrel" in df_train.columns:
        # print("indrel values ", Counter(df_train["indrel"].values))
        df_train.fillna({"indrel": 0}, inplace=True)

    # active or not
    if "ind_actividad_cliente" in df_train.columns:
        # print("ind_actividad_cliente values ", Counter(df_train["ind_actividad_cliente"].values))
        df_train.fillna({"ind_actividad_cliente": 0}, inplace=True)

    #segmentation: 01 - VIP, 02 - Individuals 03 - college graduated
    if "segmento" in df_train.columns:
        print("segmento values ", Counter(df_train["segmento"].values))
        df_train.fillna({"segmento": "03"}, inplace=True)
        df_train = label_encode(df_train, "segmento")

    #province name
    if "nomprov" in df_train.columns:
        print("nomprov values ", Counter(df_train["nomprov"].values))
        df_train.fillna({"nomprov": " "}, inplace=True)
        df_train = label_encode(df_train, "nomprov")

    #spouse index
    if "conyuemp" in df_train.columns:
        print("conyuemp values ", Counter(df_train["conyuemp"].values))
        df_train.fillna({"conyuemp": " "}, inplace=True)
        df_train = label_encode(df_train, "conyuemp")


    #for the rest fill missing values with 0
    df_train.fillna(0, inplace=True)
    print(df_train.head(10))
    return df_train


def predict_feature_prob(ids):
    """
    for each feature, take it as a label and predict it using other features' values
    """
    print("predicting feature probability")
    models = {}
    model_preds = {}
    id_preds = defaultdict(list)
    feature_index = 0
    for feature_index, feature in enumerate(item_cols[1:]):
        #don't take the user id and user features as label to predict
        y_train = df_train[feature]
        # drop this feature along axis =1
        x_train = df_train.drop([feature, 'ncodpers'], axis=1)
        # train model for this feature as label
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x_train, y_train)
        model_predict_prob = clf.predict_proba(x_train)
        p_train = model_predict_prob[:, 1]
        # accumulate the model and its prediction prob for each feature
        models[feature] = clf
        model_preds[feature] = p_train
        # for every user, add the prediction on each feature
        for id, p in zip(ids, p_train):
            #if the feature already defined for this user, the average it
            if id in id_preds and len(id_preds[id])>=feature_index+1:
                new_score = np.average(id_preds[id][feature_index], p)
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
        active = [c[0] for c in zip(item_cols[1:], row) if c[1] > 0]
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

print(df_train.dtypes)

print(df_train.head(10))

df_sample = pd.read_csv(sample_sub_file)
ids = df_train['ncodpers'].values
ids_unique = list(df_train["ncodpers"].unique())
print("total users ", len(ids), " total unique users ", len(ids_unique))

# for every feature, attempt to predict it using other features
id_preds = predict_feature_prob(ids)

used_features = get_used_features()

predict_user_items(id_preds, used_features, df_sample)