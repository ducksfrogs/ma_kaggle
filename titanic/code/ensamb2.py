import pandas as pd
import numpy as np
import re
import sklearn
import xgboost
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

import warnings

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,\
                              GradientBoostingClassifier, ExtraTreesClassifier)

from sklearn.svm import SVC
from sklearn.cross_validation import KFold

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))
plt.title('Persno Correlation of Feature', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(), linewidths=0.1, vmax=1.0,\
            square=True, cmap=colormap, linecolor='white')
