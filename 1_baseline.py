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

print(f"""
    Description
        baseline 코드 
        
    - date
        since 2021-12-04
    
    - version
        v1.0
            * notebook 생성
            * tutorial notebook 참고하여 기본 feature 생성
""")

# # 0 data load

import numpy as np
import pandas as pd
from datetime import datetime

dpath = "C:/Users/minho/kaggle dataset/g-research-crypto-forecasting/input/"
raw = pd.read_csv(dpath + 'train.csv')
raw_asset_details = pd.read_csv(dpath + 'asset_details.csv')
raw_example_test = pd.read_csv(dpath + 'example_test.csv')

raw.head(10)

raw_asset_details

# ## 1. bitcoin set

df_btc = raw[raw['Asset_ID'] == 1].set_index('timestamp')
df_btc

df_btc.info(show_counts=True)

#     비트코인 테이블 내 코인의 시작일시와 끝 일시

# +
nl = '\n'
start_btc = df_btc.index[0].astype('datetime64[s]')
end_btc = df_btc.index[-1].astype('datetime64[s]')
by_btc = df_btc.index[1] - df_btc.index[0]
length_btc = len(df_btc)

msg = f'''Bitcoin 
    start date : {start_btc}       
    end date : {end_btc}    
    by : {by_btc} seconds
    length : {length_btc}
    '''
print(msg)
# -

#     비트코인 시계열에서 index 결측치

(df_btc.index[1:]-df_btc.index[:-1]).value_counts().head()

df_btc.isna().sum()

# ## 1-1 imputation
# 첫 번째 : reindex pad
#
#     중간에 빈 시간대의 값은 직전값으로 보정해 주는 작업이다.

df_btc = df_btc.reindex(range(df_btc.index[0],
                              df_btc.index[-1] + 60, 60),
                        method = "pad")

(df_btc.index[1:]-df_btc.index[:-1]).value_counts().head()

# 총 데이터는 1956960 개임.
df_btc.info(show_counts=True)

df_btc

# * 궁금한 점 검증
#     1. 거래량이 폭증한 시간에 -> VWAP도 같이 급등한다?
#     
#        거래량 폭증 계산
#            (t-1)시점에서의 거래량보다 t시점에서 거래량 %
#            
#         
#
# * 가설
#     1. 거래량이 폭증한 시간에 그래

df_btc['diff1_Volume'] = df_btc['Volume'].diff()

df_btc














