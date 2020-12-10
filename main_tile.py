from __future__ import print_function
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from PIL import Image
from matplotlib import pyplot as plt
from sklearn import preprocessing
from PIL import Image,ImageFilter,ImageDraw,ImageEnhance, ImageChops
import random
import matplotlib.pyplot as plt
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator, load_img
from numpy import expand_dims
import scipy
from scipy.ndimage import interpolation
from numpy import asarray
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2
from cv2 import getAffineTransform
from pylab import *
import math
#from tensorflow.keras.models import load_model

import cv2
import numpy as np
import numpy as np 
import os
import tensorflow as tf
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as Keras
import h5py
import pandas
import cv2
import glob
from PIL import Image,ImageFilter,ImageDraw,ImageEnhance, ImageChops
from sklearn.model_selection import cross_val_score
from numpy.random import seed
from model_3 import *
from data_1 import *

import sklearn
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV, KFold
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import History
from matplotlib import pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
import sys
from nested_cv import NestedCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score
    

# Some image; get width and height

#image_path = 'test_set'
#image_name_arr = glob.glob(os.path.join(image_path,"*.tif"))
#image_name_arr.sort(key=lambda x: int(x.split('/')[1][:-4]))

#image = glob.glob("6465antrum_transprox_m35_s35.tif")

# Some image; get width and height
image = glob.glob("6465antrum_transprox_m27_c2.tif")
#image
image = np.uint8(io.imread(image[0]))

h, w = image.shape[1:3]

# Tile parameters
wTile = 100
hTile = 100

# Number of tiles
nTilesX = np.uint8(np.ceil(w / wTile))
nTilesY = np.uint8(np.ceil(h / hTile))

# Total remainders
remainderX = nTilesX * wTile - w
remainderY = nTilesY * hTile - h

# Set up remainders per tile
remaindersX = np.ones((nTilesX-1, 1)) * np.uint16(np.floor(remainderX / (nTilesX-1)))
remaindersY = np.ones((nTilesY-1, 1)) * np.uint16(np.floor(remainderY / (nTilesY-1)))
remaindersX[0:np.remainder(remainderX, np.uint16(nTilesX-1))] += 1
remaindersY[0:np.remainder(remainderY, np.uint16(nTilesY-1))] += 1

# Initialize array of tile boxes
tiles = np.zeros((nTilesX * nTilesY, 4), np.uint16)

# Determine proper tile boxes
k = 0
x = 0
for i in range(nTilesX):
    y = 0
    for j in range(nTilesY):
        tiles[k, :] = (x, y, hTile, wTile)
        k += 1
        if (j < (nTilesY-1)):
            y = y + hTile - remaindersY[j]

    if (i < (nTilesX-1)):
        x = x + wTile - remaindersX[i]

model = load_model('Final_model_higher_dr.hdf5')
#model = load_model('test_noise.hdf5')
size = 100
for index in range (image.shape[0]):
    tissue_img = Image.fromarray(np.uint8(image[index]))

    img_set = np.ndarray((nTilesX*nTilesY, 128,128,3), dtype="float32")
    for i in range (nTilesX*nTilesY):
        crop_size = tiles[i]
        img = tissue_img.crop((tiles[i][0],tiles[i][1],tiles[i][0]+size,tiles[i][1]+size))
        img = np.asarray(img, dtype="float32")
        img = np.reshape(img,img.shape + (1,))
        norm_image = img/255

        # image padding at four sides
        padding_size = 14
        pad_img_norm = np.pad(norm_image, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), mode='constant', constant_values=0) 
        img_set[i] = pad_img_norm

    results = model.predict(img_set,nTilesX*nTilesY ,verbose=1)

    large_img = np.zeros((512,512))
    num_pixel = np.zeros((512,512))
    for n in range(nTilesX*nTilesY):
        pre = results[n]
        pre = pre[14:114,14:114]
        start_x = tiles[n][0]
        start_y = tiles[n][1]
        for i in range (100):
            for j in range (100):
                large_img[start_y + j][start_x + i] += pre[j][i][0]
                num_pixel[start_y + j][start_x + i] += 1
        
    large_img = np.uint8(255*large_img/num_pixel)
    large_img[large_img>=127] = 255
    large_img[large_img<127] = 0
    row_img_store = 'data/img/%d.tif' % (index)
    Image.fromarray(large_img).save(row_img_store)