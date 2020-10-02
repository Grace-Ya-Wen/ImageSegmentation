import numpy as np 
import os
import tensorflow as tf
import skimage.io as io
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
from numpy.random import seed
from model_3 import *
from data_1 import *
import sklearn
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import History
from matplotlib import pyplot as plt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import sys
from sklearn.model_selection import KFold, StratifiedKFold,train_test_split

def tuning(params):
    X_train, X_test, y_train, y_test = train_test_split(params['x'], params['y'],train_size=0.8,random_state=42)
    train_img, train_mask = image_processing(X_train, y_train, target_size = (128, 128), augmentation = True, padding = True)
    validate_img, validate_mask = image_processing(X_test, y_test, target_size = (128, 128), augmentation = True, padding = True)
    
    earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    input_img = Input((128, 128, 3), name='img')
    model = get_unet(input_img, n_filters = params['n_filters'], dropout = params['dropout'], batchnorm=True)
    model.compile(optimizer=Adam(lr = params['lr']), loss="binary_crossentropy", metrics=["accuracy"])
    #model.summary()
    model.fit(train_img, train_mask,batch_size=params['batch_size'], epochs=200,shuffle = True,callbacks=[earlystop], validation_data=(X_test,y_test))
    score, acc = model.evaluate(validate_img, validate_mask, verbose=0)
    return {'loss': score, 'status': STATUS_OK, 'model': model}
