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

x, y = get_image_array('data/membrane/train/image','data/membrane/train/label', image_as_gray = False,mask_as_gray = True)

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.80, random_state=42)
#X_train_1, X_val, y_train_1, y_val = train_test_split(X_train, y_train, train_size=0.857, random_state=42)

train_img, train_mask = image_processing(X_train, y_train, target_size = (128, 128), augmentation = True, padding = True)
validate_img, validate_mask = image_processing(X_test, y_test, target_size = (128, 128), augmentation = True, padding = True)
#test_img, test_mask = image_processing(X_test, y_test, target_size = (128, 128), augmentation = True, padding = True, save = True)

# fitting parameter
#reduceLR = ReduceLROnPlateau(factor=0.05, patience=3, min_lr=0.00001, verbose=1)
model_checkpoint = ModelCheckpoint('test_full_data.hdf5', monitor='val_loss',verbose=1, save_best_only=True)
#earlystop = EarlyStopping(patience=8, verbose=1)
earlystop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=8)

# model fit
input_img = Input((128, 128, 3), name='img')
model = get_unet(input_img,n_filters=32, dropout=0.1426 , batchnorm=True)
model.compile(optimizer=Adam(8.0234e-05), loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(train_img, train_mask,batch_size=8, epochs=200, callbacks=[earlystop, model_checkpoint],shuffle = True, validation_data=(validate_img,validate_mask))


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model_3_acc_final.png')
plt.close()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model_3_loss_final.png')

# test file matrix
#test_files = glob.glob ("data/membrane/test/test_image/*.tif") 
#test = np.ndarray((len(test_files),128,128,3), dtype=np.uint8)
#test_files.sort(key=lambda x: int(x.split('/')[4][:-4]))
#for i,myFile in enumerate (test_files):
#    img = cv2.imread (myFile,cv2.IMREAD_COLOR)
#    img = np.asarray(img, dtype="float32")
#    norm_image = cv2.normalize(img,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#    pad_img_norm = np.pad(norm_image, ((14, 14), (14, 14), (0, 0)), mode='constant', constant_values=0) 
#    test[i] = pad_img_norm

#print(norm_image[15][16])
# get the test data set
#model.load_weights('padding_30batch_10epochs.hdf5') 
#model = load_model('test.hdf5')
#testGene = testGenerator("D:/deep-learning/data/membrane/test/test_padding",num_image=34)

# make prediction and save predicted images
#results = model.predict(test_img,69,verbose=1)

#saveResult("data/membrane/test/gg",results)