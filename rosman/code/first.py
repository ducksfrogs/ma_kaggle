import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pandas import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.distributions.empirical_distribution import ECDF

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
path_to = './input/'
train = pd.read_csv(path_to+'train.csv', parse_dates=True, low_memory=False, index_col='Date')
store = pd.read_csv(path_to+'store.csv', low_memory=False)

print("In total:" , train.shape)

train['Year'] = train.index.year
train['Month'] = train.index.month
train['Day'] = train.index.day
train['WeekOfYear'] = train.index.weekofyear

train['salePerCustomer'] = train['Sales'] / train['Customers']
train['salePerCustomer'].describe()

sns.set(style='ticks')
c='#386B7F'

plt.figure(figsize=(12,6))

plt.subplot(311)
cdf = ECDF(train['Sales'])
plt.plot(cdf.x, cdf.y, label='statsmodels', color=c)

plt.subplot(312)
cdf = ECDF(train['Customers'])
plt.plot(cdf.x, cdf.y, label='statsmodels', color=c)
plt.xlabel('Customers')

pl.subplot(313)
cdf = ECDF(train['salePerCustomer'])
plt.plot(cdf.x, cdf.y, label='statsmodels', color=c)
plt.xlabel('Sale Per Customer')
