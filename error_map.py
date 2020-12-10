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

# Some image; get width and height


image_path = 'error_map'
image_name_arr_1 = glob.glob(os.path.join(image_path,"46.tif"))
image_name_arr_2 = glob.glob(os.path.join(image_path,"46_predict_1.tif"))

mask_1 = io.imread(image_name_arr_1[0],as_gray = True)
mask_2 = io.imread(image_name_arr_2[0],as_gray = True)
mask_1= np.asarray(mask_1, dtype=np.uint8)
mask_2= np.asarray(mask_2, dtype=np.uint8)

error_map = np.ndarray((100,100,3), dtype=np.uint8)

for i in range (100):
    for j in range (100):
        if mask_1[i][j] == mask_2[i][j]:
            #error_map[i][j] = [0,0,0]
            if mask_1[i][j] == 0:
                error_map[i][j] = [0,0,0]
            else:
                error_map[i][j] = [255,255,255]
        else:
            if mask_1[i][j] == 0:
                error_map[i][j] = [255,0,0]
            else:
                error_map[i][j] = [0,0,255]

img = Image.fromarray(error_map, 'RGB')
img.save('46_error1.tif')
img.show()








