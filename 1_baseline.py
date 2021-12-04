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

"""
    Description
        baseline 코드 
        
    - date
        since 2021-12-04
    
    - version
        v1.0
            * notebook 생성
            * tutorial notebook 참고하여 기본 feature 생성
"""

import numpy as np
import pandas as pd
from datetime import datetime

dpath = "C:/Users/minho/kaggle dataset/g-research-crypto-forecasting/input/"
raw_train = pd.read_csv(dpath + 'train.csv')
raw_asset_details = pd.read_csv(dpath + 'asset_details.csv')
raw_example_test = pd.read_csv(dpath + 'example_test.csv')

raw_train.head(10)

raw_asset_details

# ## 1. bitcoin set

df_train_btc = raw_train[raw_train.Asset_ID == 1].set_index('timestamp')
df_train_btc.head()

df_train_btc.info()

# 비트코인 테이블 내 코인의 시작일시와 끝 일시

# +
nl = '\n'
start_btc = df_train_btc.index[0].astype('datetime64[s]')
end_btc = df_train_btc.index[-1].astype('datetime64[s]')
length_btc = len(df_train_btc)

msg = f'Bitcoin start date : {start_btc} {nl}        end date : {end_btc} {nl}        length : {length_btc}'
print(msg)
# -

# 비트코인 시계열에서 결측치

(df_train_btc.index[1:]-df_train_btc.index[:-1]).value_counts().head()
