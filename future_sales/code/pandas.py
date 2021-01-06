import numpy as np
import pandas as pd

from itertools import product
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt


from xgboost import XGBRegressor
from xgboost import plot_importance

def plot_features(booster, figsize):
    fig, ax = plt.subplots(1, 1, figsize = figsize)
    return plot_importance(booster=booster, ax=ax)

import time
import sys
import gc
import pickle

items = pd.read_csv('../input/items.csv')
shops = pd.read_csv('../input/items.csv')
cats = pd.read_csv('../input/item_categories.csv')
train = pd.read_csv('../input/sales_train.csv')
test = pd.read_csv('../input/test.csv').set_index('ID')


train = train[train.item_price < 10000]
train = train[train.item_cnt_day < 1001]

median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)\
               &(train.item_price > 0)].item_price.median()
train.loc[train.item_price < 0, 'item_price'] = median
