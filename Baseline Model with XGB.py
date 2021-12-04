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

# +
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc

from xgboost import XGBRegressor
import gresearch_crypto
env = gresearch_crypto.make_env()
iter_test = env.iter_test()


# +
def add_time_features(df):
    df = df.assign(ds = df.index.values.astype('datetime64[s]'))
    
    df = df.assign(month = df['ds'].dt.month)
    df = df.assign(dayofweek = df['ds'].dt.dayofweek)
    df = df.assign(dayofmonth = df['ds'].dt.day)
    df = df.assign(dayofyear = df['ds'].dt.dayofyear)
    
    df = df.assign(hour = df['ds'].dt.hour)
    df = df.assign(minute = df['ds'].dt.minute)
    
    return df

def plot_ts(df, c='Target', n_cols = 4):
    assets = df.Asset_ID.unique()
    n_rows = int(np.ceil(len(assets) / n_cols))
    f, ax = plt.subplots(n_rows, n_cols, figsize=(20,10))
    i, j = 0, 0
    for asset in assets:
        asset_name = asset_details_df[asset_details_df.Asset_ID == asset].Asset_Name.iloc[0]
        sub_df = df[df.Asset_ID == asset]
        ax[i][j].plot(sub_df[c], label=asset_name)
        ax[i][j].legend()
        
        j += 1
        j = j % n_cols
        if j == 0:
            i += 1


# -

folder = "C:/Users/minho/kaggle dataset/g-research-crypto-forecasting/input/"
train_df = pd.read_csv(folder + 'train.csv')
asset_details_df = pd.read_csv(folder + 'asset_details.csv')
example_test_df = pd.read_csv(folder + 'example_test.csv')

example_test_df.head(1)

cutoff_ds = '2021-09-01'
train_df = train_df.set_index('timestamp')
train_df = train_df.assign(ds=train_df.index.values.astype('datetime64[s]'))
train_df = train_df[train_df.ds > cutoff_ds]
train_df = add_time_features(train_df)

# +
X = train_df[(~train_df.Target.isna()) & (train_df.VWAP != np.float('inf'))].drop(['ds', 'Target'], axis = 1)
y = train_df[(~train_df.Target.isna()) & (train_df.VWAP != np.float('inf'))]['Target']

model = XGBRegressor()
model.fit(X, y)
# -

for (test_df, sample_prediction) in iter_test:
    test_df = test_df.set_index('timestamp')
    test_df = test_df.assign(ds = test_df.index.values.astype('datetime64[s]'))
    test_df = add_time_features(test_df)
    sample_prediction_df['Target'] = model.predict(test_df.drop(['ds', 'row_id'], axis = 1))
    env.predict(sample_prediction_df)






















