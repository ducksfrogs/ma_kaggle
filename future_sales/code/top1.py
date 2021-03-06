import numpy as np
import pandas as pd
import random as rd
import datetime

import matplotlib.pyplot as plt
import seaborn as sns


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace import SARIMAX
from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.stattools import adfuller, acf, pacf, arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs

import warnings
warnings.filterwarnings('ignore')

sales = pd.read_csv("../input/sales_train.csv")

item_cat = pd.read_csv("../input/item_categories.csv")
item = pd.read_csv("../input/items.csv")
sub = pd.read_csv("../input/sample_submission.csv")
shops = pd.read_csv("../input/shops.csv")
test = pd.read_csv("../input/test.csv")


sales.date = sales.date.apply(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y'))


monthly_sales = sales.groupby(['date_block_num', 'shop_id', 'item_id'])[ 'date','item_price',
                                'item_cnt_day'].agg('date':['min', 'max'], 'item_price':'mean','item_cnt_day':'sum'))


x = item.groupby(['item_category_id']).count()
x = x.sort_values(by='item_id', ascending=False)
x = x.iloc[0:10].reset_index()


plt.figure(figsize=(8,4))
ax = sns.barplot(x.item_category_id, x.item_id, alpha=0.8)
plt.title("Items per Category")
plt.ylabel("# of items", fontsize=12)
plt.xlabel("Category", fontsize=12)
plt.show()

ts = sales.groupby(['date_block_num'])['item_cnt_day'].sum()
ts.astype('float')
plt.figure(figsiz=(16,8))
plt.title("Total Sales of the company")
plt.xlabel("Time")
plt.ylabel("Sales")
plt.plot(ts)

plt.figure(figsize=(16,6))
plt.plot(ts.rolling(window=12, center=False).mean(), label='Rolling Mean')
plt.plot(ts.rolling(window=12, center=False).std(), label='Rolling sd')
plt.legend()

import statsmodels.api as sm

res = sm.tsa.seasonal_decompose(ts.values, freq=12, model='mulitplicative')
