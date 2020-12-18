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


#wave library

import wave
wav = wave.open(fname)
print("Sampling (frame) rate = ", wav.getframerate())
print("Total samples (frames) = ", wav.getnframes())
print("Duration = ", wav.getnframes()/wav.getframerate())


#Using scipy
from scipy.io import wavfile
rate, data = wavfile.read(fname)
print("Sampling (frame) rate = ", rate)
print("Total samples (frames) = ", data.shape)
print(data)

plt.plot(data, '-', )

plt.figure(figsize=(16,4))
plt.plot(data[:500], '-'); plt.plot(data[:500], '-'); plt

train['nframes'] = train['fname'].apply(lambda f: wave.open('../input/train_curated/' + f).getnframes())
test['nframes'] = test['fname'].apply(lambda f: wave.open('../input/test/' + f).getnframes())
