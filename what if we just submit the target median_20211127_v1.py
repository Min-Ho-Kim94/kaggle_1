# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: cook
#     language: python
#     name: cook
# ---

import datetime
import traceback
import numpy as np
import pandas as pd
import xgboost as xgb

data_folder = "C:/Users/minho/kaggle dataset/g-research-crypto-forecasting/input/"
df = pd.read_csv(data_folder + 'train.csv')
df_asset_details = pd.read_csv(data_folder + 'asset_details.csv')
df.shape, df.info

asset_to_weight = df_asset_details.Weight.values
asset_to_weight

# apply(pd.Series, axis=0)
#
#     apply함수는 Series를 객체로 받는다.
#     axis=0이면 열에 대해 계산을
#     axis=1이면 행에 대한 계산을 수행한다.

df['Weight'] = df['Asset_ID'].apply(lambda x: asset_to_weight[x])
# type(df['Asset_ID'])

asset_to_weight[4]


# +
def clean(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True) #????
    df.dropna(how='any', inplace=True)
    
def test_train_split(df):
    X_train = df[df['timestamp'] <= 1623542400].drop('Target', axis=1)
    y_train = df[df['timestamp'] <= 1623542400].Target
    X_test = df[df['timestamp'] > 1623542400].iloc[:-1].drop('Target', axis=1) # 마지막 행을 제거한다.
    y_test = df[df['timestamp'] > 1623542400].iloc[:-1].Target
    return X_train, y_train, X_test, y_test

clean(df)
df = df.loc[pd.to_datetime(df['timestamp'], unit='s').dt.year==2021]

X_train, y_train, X_test, y_test = test_train_split(df)
# -

(df.replace([np.inf, -np.inf], np.nan, inplace=True) is df)
pd.to_datetime(df['timestamp'], unit='s').dt.year

raw_data = {'col0': [1, 2, 3, 4],
            'col1': [10, 20, 30, 40],
            'col2': [100, 200, 300, 400]}
data = pd.DataFrame(raw_data)
data, data.iloc[:-1]

X_train.head()


class StupidMean:
    def __init__(self):
        self.target_median = None # attribute
        
    def fit(self,X_train, y_train):
        self.target_median = y_train.median()
        
    def predict(self, X_test):
        return self.target_median 


features = ['Count', 'Open', 'Close', 'High', 'Low', 'Volume']

# +
models = [StupidMean() for _ in range(len(df_asset_details))] # _: index가 for문 내에서 굳이 필요하지 않을 경우 사용함.
models # list type

y_pred = pd.Series(data=np.full_like(y_test.values, np.nan), index=y_test.index) #np.full_like(벡터(array, list, tuple), 채워야하는 값)
for asset_ID, model in enumerate(models): #enumerate() : enumerate 객체 반환. 인덱스와 value 값을 포함하는.
    X_asset_train = X_train[X_train.Asset_ID==asset_ID]
    y_asset_train = y_train[X_train.Asset_ID==asset_ID]
    X_asset_test = X_test[X_test.Asset_ID==asset_ID]
    
    model.fit(X_asset_train[features], y_asset_train)
    y_pred[X_test.Asset_ID == asset_ID] = model.predict(X_asset_test[features])
    print(f'Trained model for asset {asset_ID}')


# +
def corr(a, b, w):
    cov = lambda x, y: np.sum(w * (x - np.average(x, weights=w)) * (y - np.average(y, weights=w))) / np.sum(w)
    return cov(a, b) / np.sqrt(cov(a, a) * cov(b, b))

score = corr(y_pred, y_test.values, X_test.Weight)
print(f'{score:.5f}')
# -

y_pred

print(models[0].target_median)








