import os, cv2, random

import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_image
from keras import layers, models, optimizers
from keras import backend as K
from sklearn.model_selection import train_test_split

img_width = 150
img_height = 150
TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'
train_images_dogs_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
test_images_dogs_cats = [TEST_DIR+i for i in os.listdir(TEST_DIR)]


def atoi(text):
    return int(text) if text.isdigit() else text

def naturaL_keys(text:
        return [ atoi(c) for c in re.split('(\d+)', text)]


train_images_dogs_cats.sort(key=naturaL_keys)
train_images_dogs_cats = train_images_dogs_cats[0:1300] + train_images_dogs_cats[12500:13000]

test_images_dogs_cats.sort(key=naturaL_keys)
