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

# K-fold Cross Validation model evaluation
num_folds = 5

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)
acc_per_fold = []
loss_per_fold = []
parameter_set = []

# parameters
fold_no = 1
batch_size = [8, 16, 32]
n_filters = [16, 32, 64]

# read image data as array
x, y = get_image_array('data/membrane/train/image','data/membrane/train/label', image_as_gray = False,mask_as_gray = True)

# tuning process
for train, test in kfold.split(x, y):
    # define hyperparameter space
    trials = Trials()
    params = {'batch_size': hp.choice('batch_size',[8, 16, 32]), 
          'dropout':hp.uniform('dropout', .05,.4),
          'lr':hp.uniform('lr', 1e-6,1e-4),
          'n_filters': hp.choice('n_filters',[16, 32, 64]),
          'x':x[train],
          'y':y[train]}

    best = fmin(tuning,params, algo=tpe.suggest, max_evals=20, trials=trials)
    parameter_set.append(best)

    # fit model with the best hyper parameter set
    input_img = Input((128, 128, 3), name='img')
    model = get_unet(input_img,n_filters=n_filters[best['n_filters']], dropout=best['dropout'], batchnorm=True)
    model.compile(optimizer=Adam(best['lr']), loss="binary_crossentropy", metrics=["accuracy"])
    earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    train_img, train_mask = image_processing(x[train], y[train], target_size = (128, 128), augmentation = True, padding = True)
    test_img, test_mask = image_processing(x[test], y[test], target_size = (128, 128), augmentation = True, padding = True)
    model.fit(train_img, train_mask,batch_size=batch_size[best['batch_size']], epochs=200,shuffle = True,callbacks=[earlystop],validation_data=(test_img, test_mask))
    # Calculate the score
    scores = model.evaluate(test_img, test_img, verbose=0)
    acc_per_fold.append(scores[1])
    loss_per_fold.append(scores[0])


print(parameter_set)
print(acc_per_fold)
print(loss_per_fold)
