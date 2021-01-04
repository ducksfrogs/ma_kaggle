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

PassengerId = test['PassengerId']

full_data = [train, test]

train['Name_length'] = train['Name'].apply(len)
test['Name_length'] = test['Name'].apply(len)

train['Has_Cabin'] = train['Cabin'].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test['Cabin'].apply(lambda x: 0 if type(x) == float else 1 )

for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize']==1, 'IsAlone'] = 1

for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('$')

for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

train['CategoricalFare'] = pd.qcut(train['Fare'],4)

for dataset in full_data:
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_int = np.random.randint(age_avg-std, age_avg+age_std,size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_int
    dataset['Age'] = dataset['Age'].astype(int)

train['CategoricalAge'] = pd.cut(train['Age'], 5)


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.' name)
    if title_search:
        return title_search.group(1)

for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)


for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt',
                                                 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkeer','Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', "Mrs")

for dataset in full_data:
    dataset["Sex"] = dataset['Sex'].map({ 'femail': 0, 'male': 1}).astype(int)
    title_mapping = {'Mr':1, 'Miss': 2, "Mrs":3, 'Master': 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C': 1, 'Q':2}).astype(int)
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[ (dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare' ] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[ (dataset['Fare'] > 31), 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age']> 16)&(dataset['Age']<=32), 'Age'] = 1
    dataset.loc[(dataset['Age']>32)&(dataset['Age'] <= 48), 'Age'] =2
    dataset.loc[(dataset['Age']> 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age' ] = 4

    drop_element = ['PassengerId', 'Name', 'Ticket', 'Cabin','SibSp']
    train = train.drop(drop_element, axis=1)
    train = train.drop(['CategoricalAge', 'CategricalFare'], axis=1)
    test = test.drop(drop_element, axis=1)



colormap = plt.cm.RdBu

plt.figure(figsize=(14,12))
plt.title('Persno Correlation of Feature', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(), linewidths=0.1, vmax=1.0,\
            square=True, cmap=colormap, linecolor='white')

g = sns.pairplot(train[[u'Survived', u'Pclass', u'Sex', u'Age',
                        u'Parch', u'Fare', u'Embarked', u'FamilySize', u'Title']],
                 hue='Survived', palette='seismic', size=1.2, diag_kind='kde',
                 diag_kws=dict(shade=True), plot_kws=dict(s=10))
g.set(xticklabels=[])


ntrain = train.shape[0]
ntest = test.shape[0]

SEED = 0
NFOLDS = 5
kf = KFold(ntrain, n_folds=NFOLDS, random_state=SEED)


class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)

    def fit(self, x, y):
        return self.clf.fit(x,y)

    def feature_importances(self, x, y):
        print(self.clf.fit(x,y).feature_importances_)
