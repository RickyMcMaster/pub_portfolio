# basic
import pandas as pd
import pickle

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

"""

PROCESS FLOW

* feature_engineering
* split_data
* feature_dict
* vectorizer
* train
* model.predict()
"""

df = pd.read_csv("train.csv")


def feature_engineering(df):
    df.drop("Id", axis=1, inplace=True)
    df["LotFrontage"] = df["LotFrontage"].fillna(0)
    numeric = list(df.select_dtypes("number").columns)
    categorical = list(df.select_dtypes("object").columns)
    # Remove target
    numeric.remove("SalePrice")
    # Remove features
    for c in ["MSSubClass", "YrSold"]:
        df[c] = df[c].astype(str)
    numeric.remove(c)
    categorical.append(c)
    for c in [
        "Street",
        "Alley",
        "LandContour",
        "Utilities",
        "LandSlope",
        "Condition2",
        "RoofMatl",
        "Heating",
        "Electrical",
        "GarageQual",
        "GarageCond",
        "PavedDrive",
        "PoolQC",
        "MiscFeature",
        "BsmtCond",
        "Functional",
        "CentralAir",
    ]:
        categorical.remove(c)
    return df, numeric, categorical


df, numeric, categorical = feature_engineering(df)


def split_data(df, target):
    df_train_full, df_test = train_test_split(
        df[numeric + categorical + [target]], test_size=0.2, random_state=1
    )
    df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=1)

    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values
    for d in [df_train, df_val, df_test]:
        del d[target]
    return df_train, df_val, df_test, y_train, y_val, y_test


df_train, df_val, df_test, y_train, y_val, y_test = split_data(df, "SalePrice")


def check_splits(df_train, df_val, df_test, y_train, y_val, y_test):
    print(len(df_train), len(df_val), len(df_test))
    print(len(df_train) + len(df_val) + len(df_test) == len(df))
    print(y_train[:25])
    print(y_val[:25])
    print(y_test[:25])


check_splits(df_train, df_val, df_test, y_train, y_val, y_test)


def feature_dict(data):
    data = data.fillna(0)
    feat_dict = data.to_dict(orient="records")
    return feat_dict


train_dict = feature_dict(df_train)
val_dict = feature_dict(df_val)
test_dict = feature_dict(df_test)


def vectorizer(train_dict):
    """Takes training feature dictionaries and vectorizes them for model training."""
    dv = DictVectorizer(sparse=False)
    dv.fit(train_dict)
    return dv


dv = vectorizer(train_dict)

X_train = dv.transform(train_dict)
X_val = dv.transform(val_dict)
X_test = dv.transform(test_dict)


def train(X, y, model_type):
    """Takes vectorized feature list plus training target and returns trained model"""
    print("Training Model...")
    model = model_type
    model.fit(X, y_train)
    print("Model Complete!")
    return model


model = train(X_train, y_train, Ridge(alpha=10, solver="svd"))


print("Saving output to pickle")
with open("houses-model.bin", "wb") as f_out:
    pickle.dump((dv, model), f_out)
