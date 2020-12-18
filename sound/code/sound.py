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
