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

def accuracy(predict,truth):
    w, h = predict.shape
    count = 0
    for i in range(w):
        for j in range(h):
            if predict[i][j] == truth[i][j]:
                count+=1
    acc = count/(w*h)
    return acc


predict_files = glob.glob ("/hpc/gwen488/DeepLearning/data/membrane/test/0.01/*.tif")
#print(len(predict_files))
#print(predict_files)
#predict_files.sort(key=lambda x: int(x.split('\\')[1][0:1]))
predict_files.sort(key=lambda x: int(x.split('_')[0][50:]))
#print(predict_files)
#predict_files.sort(key=lambda x: int(x.split('/')[1][:-12]))
predict = np.ndarray((len(predict_files),100,100), dtype=np.uint8)
for i,myFile in enumerate (predict_files):
    image = cv2.imread (myFile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.asarray(image, dtype="int32")
    #image = image.reshape(image.shape+(1,))
    predict[i] = image


files = glob.glob ("/hpc/gwen488/DeepLearning/data/membrane/test/gold/*.tif")
#files.sort(key=lambda x: int(x.split('.')[0]))
files.sort(key=lambda x: int(x.split('.')[0][-1]))
acc = []

for i in range(len(files)):
    #img = predict[i].reshape(256,256)
    #img = np.uint8(255*img)
    truth = cv2.imread (files[i])
    truth = cv2.cvtColor(truth, cv2.COLOR_BGR2GRAY)
    acc.append(accuracy(predict[i],truth))

print(sum(acc)/len(acc))
print(acc)