import numpy as np
import pandas as pd
import random as rd
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf, arma_order_select_ic
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as stats

import warnings

warnings.filterwarnings('ignore')

sales = pd.read_csv("../input/sales_train.csv")
item_cat = pd.read_csv("../input/item_categories.csv")
item = pd.read_csv('../input/items.csv')
sub = pd.read_csv("../input/sample_submission.csv")
shops = pd.read_csv("../input/shops.csv")
test = pd.read_csv("../input/test.csv")

sales.date = sales.date.apply(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y'))

print(sales.info())

manthly_sales = sales.groupby(['date_block_num','shop_id'])['date','item_price','item_cnt_day']\
                                .agg({'date':['min','max'], 'item_price':'mean', 'item_cnt_day':'sum'})
