import numpy as np
import pandas as pd
from itertools import product
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt

def plot_features(booster, figsize):
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)


import time
import sys
import gc
import pickle

items = pd.read_csv("./in")
shops = pd.read_csv()
cats = pd.read_csv()
train = pd.read_csv()

test = pd.read_csv()
