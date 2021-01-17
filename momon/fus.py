import numpy as np
import pandas as pd

from itertools import product
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt

def plot_feature(booster, figsize):
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

import time
import sys
import gc
import pickle

item = pd.read_csv('../input/items.csv')
shops = pd.read_csv('../input/shops.csv')
cats = pd.read_csv('../input/cats.csv')
train = pd.read_csv('../input/sales_train.csv')

test = pd.read_csv("../input/test.csv")

plt.figure(figsize=(10,4))
plt.xlim(-100, 3000)
sns.boxplot(x=train.item_cnt_day)
plt.figure(figsize=(10,4))
plt.xlim(train.item_price.min(), train.item_price.max*1.1)
sns.boxplot(x=train.item_price)

train = train[train.item_price<10000]
train = train[train.item_cnt_day<1001]

median = train[(train.shop_id==32)&
               (train.item_id==2973)&
               (train.date_block_num==4)&
               (train.item_price>0)].item_price.median()

train.loc[train.item_price<0, 'item_price'] = median


len(list(set(test.item_id)- set(set.item_id).intersection(set(train.item_id)))), len(list(set(test.item_id))), len(test)
