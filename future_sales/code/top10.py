import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import time

from math import sqrt
from numpy import loadtxt
from itertools import product
from sklearn import preprocessing
from xgboost import plot_tree
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import  TfidfVectorizer


kernel_with_output = False

if kernel_with_output:
    sales_train = pd.read_csv('../input/sales_train.csv')
    items = pd.read_csv('../input/items.csv')
    shops = pd.read_csv("../input/shops.csv")
    item_categories = pd.read_csv('../input/item_categories.csv')
    test =pd.read_csv('../input/test.csv')
    sample_submission = pd.read_csv('../input/sample_submission.csv')


if kernel_with_output:
    grid = []
    for block_num in sales_train['date_block_num'].unique():
        cur_shops = sales_train[sales_train['date_block_num']==block_num]['shop_id'].unique()
        cur_items = sales_train[sales_train['date_block_num'] == block_num]['item_id'].unique()
        grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))
    index_cols = ['shop_id', 'item_id', 'date_block_num']
    grid = pd.DataFrame(np.vstack(grid), columns=index_cols, dtype=np.int32)
    sales_train['item_cnt_day'] = sales_train['item_cnt_day'].clip(0,20)
    groups = sales_train.groupby(['shop_id', 'item_id','date_block_num'])
    trainset = groups.agg({'item_cnt_day':'sum', 'item_price':'mean'}).reset_index()

    trainset = trainset.rename(columns={'item_cnt_day':'item_cnt_month'})
    trainset['item_cnt_month'] = trainset['item_cnt_month'].clip(0,20)
    trainset = pd.merge(trainset, items[['item_id','item_category_id']], on='item_id')
    trainset.to_csv('trainset_with_grid.csv')

    trainset.head()

if kernel_with_output:
    prev_month_selector = (trainset.month == 10) & (trainset.year == 2015)
    train_subset = trainset[prev_month_selector]
    groups = train_subset[['shop_id','item_id','item_cnt_month']].groupby(by=['shop_id','item_id'])
    train_subset = groups.agg({'item_cnt_month':'sum'}).reset_index()
    train_subset.head()


if kernel_with_output:
    maeged = test.merge(trainset, on=['shop_id','item_id'], how='left')[["ID","item_cnt_month"]]
    merged.isna().sum()

if kernel_with_output:
    merged['item_cnt_month'] = merged.item_cnt_month.fillna(0).clip(0, 20)
    submission = merged.set_index("ID")
    submission.to_csv("benchmark.csv")


if kernel_with_output:
    baseline_features = ['shop_id', 'item_id', 'item_category_id', 'date_block_num', 'item_cnt_month']
    train = trainset[baseline_features]
    train = train.set_index('shop_id')