import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):
    pass

warnings.warn = ignore_warn

from scipy import stats
from scipy.stats import norm, skew

pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


print("the train data size before dropping Id feature is : {}".format(train.shape))
print("the test data size before dropping Id feature is : {}".format(test.shape))

train_ID = train["Id"]
test_ID = test["Id"]

train.drop('Id', axis=1, inplace = True)
test.drop("Id", axis=1, inplace = True )

print("\nTeh train data size after dropping Id feature is :{}".format(train.shape))
print("\nthe test data size after dropping Id feature is : {}".format(test.shape))


fig, ax = plt.subplots()
ax.scatter(x=train['GrLivArea'], y=train['SalePrice'])
plt.ylabel("SalePrice", fontsize=13)
plt.xlabel("GrLivArea", fontsize=14)

train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)


sns.distplot(train['SalePrice'], fit=norm)

(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)],
            loc='best')

plt.ylabel('Frequency')
plt.title("SalePrice distribution")

fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)


train['SalePrice'] = np.log1p(train['SalePrice'])
sns.distplot(train['SalePrice'], fit=norm)
(mu, sigma) = norm.fit(train['SalePrice'])
print('\n mu = {:.2f} and {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title("SalePrice distribution")

fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)


ntrain = train.shape[0]
ntest = test.shape[0]

y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))

all_data_na = (all_data.isnull().sum() / len(all_data)) *100
all_data_na = all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending=False)[:30]

missing_data = pd.DataFrame({ 'Missing Ratio': all_data_na})
