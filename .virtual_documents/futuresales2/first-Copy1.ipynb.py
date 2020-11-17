import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)


from itertools import product
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt

import time
import sys
import gc

import pickle

items = pd.read_csv('../input/items.csv')
shops = pd.read_csv('../input/shops.csv')
cats = pd.read_csv('../input//item_categories.csv')
train = pd.read_csv('../input/sales_train.csv')

test = pd.read_csv('../input/test.csv')



train = train[train.item_price < 100000]
train = train[train.item_cnt_day < 1001]

matrix = []

cols = ['date_block_num', 'shop_id', 'item_id']



for i in range(34):
    sales = train[train.date_block_num==i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))



for i in range(2):
    sales = train[train.date_block_num==i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))



matrix 



