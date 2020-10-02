from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
from PIL import Image,ImageFilter,ImageDraw,ImageEnhance, ImageChops
import random
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.ndimage import interpolation
from numpy import asarray
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import cv2

def compute_iou(y_pred, y_true):
    # ytrue, ypred is a flatten vector
    y_pred = np.rint(y_pred.flatten())
    y_true = np.rint(y_true.flatten())
    current = confusion_matrix(y_true, y_pred, labels=[1, 0])
    # compute mean iou
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    return np.mean(IoU)


def sensitivity(y_pred, y_true):
    # ytrue, ypred is a flatten vector
    y_pred = np.rint(y_pred.flatten())
    y_true = np.rint(y_true.flatten())
    cm1 = confusion_matrix(y_true, y_pred, labels=[1, 0])
    sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    return sensitivity1

def specificity(y_pred, y_true):
    # ytrue, ypred is a flatten vector
    y_pred = np.rint(y_pred.flatten())
    y_true = np.rint(y_true.flatten())
    cm1 = confusion_matrix(y_true, y_pred, labels=[1, 0])
    specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    return specificity1

def accuracy(predict,truth):
    w, h = predict.shape
    count = 0
    for i in range(w):
        for j in range(h):
            if predict[i][j] == truth[i][j]:
                count+=1
    acc = count/(w*h)
    return acc

def DSC(predict, truth):

    smooth = 1 
    #y_true_f = K.flatten(truth) # flatten to 1D
    #y_pred_f = K.flatten(predict)
    intersection = K.sum(truth * predict)
    union = K.sum(truth) + K.sum(predict)
    #intersection = K.sum(y_true_f * y_pred_f)
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


def IoU(predict, truth):
    w, h = predict.shape
    inter = 0
    union = 0

    for i in range(w):
        for j in range(h):
            if truth[i][j] == 0 and predict[i][j] == 0:
                inter+=1
            if truth[i][j] == 0 or predict[i][j] == 0:
                union+=1
    return inter/union


predict_files = glob.glob ("you_path_stored_the_prediction")

predict = np.ndarray((len(predict_files),100,100), dtype=np.uint8)
for i,myFile in enumerate (predict_files):
    image = cv2.imread (myFile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.asarray(image, dtype="int32")
    predict[i] = image

files = glob.glob ("you_path_stored_the_gold_standards")

acc = []
for i in range(len(files)):
    #img = predict[i].reshape(256,256)
    #img = np.uint8(255*img)
    truth = cv2.imread (files[i])
    truth = cv2.cvtColor(truth, cv2.COLOR_BGR2GRAY)
    acc.append(accuracy(predict[i],truth))

print(sum(acc)/len(acc))
print(acc)
