import numpy as np
import pandas as pd
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
combined_df = [train_df, test_df]

train_df.info()
print("_"*50)
test_df.info()

train_df.describe()

train_df.describe(include=['O'])


train_df[['Pclass','Survived']].groupby(['Pclass'], as_index=False).mean()\
            .sort_values(by='Survived')

train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

train_df[['Parch','Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, "Age", bins=20)

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2,\
                      aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

grid = sns.FacetGrid(train_df, row='Embarked', col='Survived',size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


#Correcting by dropping features

print("Before ", train_df.shape, test_df.shape, combined_df[0].shape, combined_df[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(["Ticket",'Cabin'], axis=1)
combined_df = [train_df, test_df]

for dataset in combined_df:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])


for dataset in combined_df:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', "Capt"\
                                                 , "Col", 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer'\
                                                 'Dona'], "Rare")

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {'Mr':1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}

for dataset in combined_df:
    dataset['Title'] = dataset['Title'].replace(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
