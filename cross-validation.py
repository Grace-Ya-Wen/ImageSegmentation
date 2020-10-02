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
#files = glob.glob ("data/membrane/train/test1/*.tif") 
#x_train = np.ndarray((len(files),128,128,3), dtype=np.uint8)

#files.sort(key=lambda x: int(x.split('/')[4][:-4]))
#print(files)
#for i,myFile in enumerate (files):
#    image = cv2.imread (myFile,cv2.IMREAD_COLOR)
#    image = np.asarray(image, dtype="int32" )
#    norm = (image - np.min(image)) / (np.max(image) - np.min(image))
#    x_train[i] = norm

# training mask data
#files = glob.glob ("data/membrane/train/test2/*.tif")
#y_train = np.ndarray((len(files),128,128,1), dtype=np.uint8)
#files.sort(key=lambda x: int(x.split('/')[4][:-4]))
#for i,myFile in enumerate (files):
#    image = cv2.imread (myFile)
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    image = np.asarray(image, dtype="int32" )
#    image = image.reshape(image.shape+(1,))
#    norm = image/255
#    y_train[i] = norm


# read image data as array
x, y = get_image_array('data/membrane/train/image','data/membrane/train/label', image_as_gray = False,mask_as_gray = True)

num_folds = 5
acc_per_fold = []
loss_per_fold = []

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1

# fitting parameter
for train, test in kfold.split(x, y):
    
    train_img, train_mask = image_processing(x[train], y[train], target_size = (128, 128), augmentation = True, padding = True)
    test_img, test_mask = image_processing(x[test], y[test], target_size = (128, 128), augmentation = True, padding = True)
    earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    # model fit
    input_img = Input((128, 128, 3), name='img')
    model = get_unet(input_img,n_filters=32, dropout=0.14256, batchnorm=True)
    model.compile(optimizer=Adam(7.006e-05), loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(train_img, train_mask,batch_size=8, epochs=200,shuffle = True,callbacks=[earlystop], validation_data = (test_img,test_mask))
    
    # Generate generalization metrics
    scores = model.evaluate(test_img, test_mask, verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])
    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')
