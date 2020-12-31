import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from collections import Counter

from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                            GradientBoostingClassifier, ExtraTreesClassifier,
                            VotingClassifier)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
IDtest = test['PassengerId']

def detect_outliners(df, n, features):
    outlier_indices = []

    for col in features:
        Q1 = np.percentile(df[col], 25)

        Q3 = np.percentile(df[col], 75)

        IQR = Q3 - Q1

        outlier_step = 1.5 * IQR

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index

        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list( k for k, v  in outlier_indices.items() if v > n)

    return multiple_outliers


Outliers_to_drop = detect_outliners(train, 2, ['Age', 'SibSp','Parch','Fare'])

train.loc[Outliers_to_drop]
