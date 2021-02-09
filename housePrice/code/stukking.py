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
