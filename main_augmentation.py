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


# training data
x, y = get_image_array('data/membrane/train/image','data/membrane/train/label', image_as_gray = False,mask_as_gray = True)
train_img, train_mask = image_processing(x, y, target_size = (128, 128), augmentation = True, padding = True)

# fitting parameter
#reduceLR = ReduceLROnPlateau(factor=0.05, patience=3, min_lr=0.00001, verbose=1)
model_checkpoint = ModelCheckpoint('test.hdf5', monitor='loss',verbose=1, save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

# model fit
input_img = Input((128, 128, 3), name='img')
model = get_unet(input_img,n_filters=32, dropout=0.14256, batchnorm=True)
model.compile(optimizer=Adam(7.006e-05), loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train,batch_size=8, epochs=10, callbacks=[model_checkpoint,earlystop],shuffle = True)

# test file matrix
test_files = glob.glob ("data/membrane/test/test_raw/*.tif") 
test = np.ndarray((len(test_files),128,128,3), dtype=np.uint8)
test_files.sort(key=lambda x: int(x.split('/')[4][:-4]))
for i,myFile in enumerate (test_files):
    image = cv2.imread (myFile,cv2.IMREAD_COLOR)
    image = np.asarray(image, dtype="int32")
    norm = (image - np.min(image)) / (np.max(image) - np.min(image))
    test[i] = image


# get the test data set
#model.load_weights('padding_30batch_10epochs.hdf5') 
#model = load_model('model_3with_best_parameter.hdf5')
#testGene = testGenerator("D:/deep-learning/data/membrane/test/test_padding",num_image=34)

# make prediction and save predicted images
results = model.predict(test,84,verbose=0)

saveResult("data/membrane/test/da",results)
