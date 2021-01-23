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
