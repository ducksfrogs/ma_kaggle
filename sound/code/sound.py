import numpy as np
np.random.seed(1001)

import os
import shutil
import warnings

import IPython
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm_notebook
from sklearn.model_selection import StratifiedKFold

matplotlib.style.use('ggplot')
warnings.filterwarnings('ignore', category=FutureWarning)


os.listdir('../input')

train = pd.read_csv("../input/train_curated.csv")
test = pd.read_csv("../input/sample_submission.csv")

train.sample(10)
test.sample(5)

print("Number of train examples=", train.shape[0], "Number of classes", len(set(train.labels)))
print("Number of test examples=", test.shape[0], "number of classes=", len(set(test.columns[1:])))


train = train[train.labels.isin(test.columns[1:])]
print(len(train))

category_group = train.groupby(['labels']).count()
category_group.columns = ['counts']
print(len(category_group))


plot = category_group.sort_values(ascending=True, by='counts').plot(
    kind='barh',
    title='number of Audio samples per category',
    color='deeppink',figsize=(15,20)
)

plot.set_xlabel('category')
plot.set_ylabel("Number of Samples")
