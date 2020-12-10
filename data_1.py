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
#import tensorflow as tf

def adjustData(img,mask, data_augmentation = True, padding = True,target_size = (128,128), save = False):

    if data_augmentation == True:
        ED_img, ED_mask = create_ED_dataset(img, mask, img_nums = 4, save = save)    # elastic transformation (4 images)
        flip_img, flip_mask = create_flip_set(img, mask, save = save)                # flip (7 images)
        rotate_img, rotate_mask = create_rotation_set(img, mask, save = save)        # rotation&crop (6 images)
        crop_img, crop_mask = crop_image_set(img, mask, save = save)                 # crop (5 images)
        img_set = np.vstack((ED_img,flip_img,rotate_img,crop_img))
        mask_set = np.vstack((ED_mask,flip_mask,rotate_mask,crop_mask))
    else:
        img_set = np.asarray(img).reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        mask_set = np.asarray(mask).reshape((1, mask.shape[0], mask.shape[1], mask.shape[2]))
        
        if save == True:
            uniq = 0
            row_img_store = 'data/membrane/test/test_image/%d.tif' % (uniq)
            label_img_store = 'data/membrane/test/test_label/%d.tif' % (uniq)
            while os.path.exists(row_img_store):
                row_img_store = 'data/membrane/test/test_image/%d.tif' % (uniq)
                label_img_store = 'data/membrane/test/test_label/%d.tif' % ( uniq)
                uniq += 1
            Image.fromarray(img).save(row_img_store)
            Image.fromarray(mask[:,:,0]).save(label_img_store)

    # image transfer to the target size 
    img_set_1 = np.ndarray((img_set.shape[0], target_size[0],target_size[1],3), dtype="float32")
    mask_set_1 = np.ndarray((mask_set.shape[0], target_size[0],target_size[1],1), dtype="float32")

    for i in range (img_set.shape[0]):
        image = np.asarray(img_set[i,:,:], dtype="float32")
        mask = np.asarray(mask_set[i,:,:], dtype="float32")

        # image normalisation
        #norm = (image - np.min(image)) * (1.0/(np.max(image) - np.min(image)))
        norm_image = cv2.normalize(image,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #norm_image = tf.image.per_image_standardization(image)
        #with tf.compat.v1.Session() as sess:
        #norm_image = sess.run(std_image)
            #print(norm_image)
        norm_mask = mask* (1.0/255)

        # image padding at four sides
        padding_size = int(0.5*(target_size[0] - img.shape[1]))
        if (padding == True):
            pad_img_norm = np.pad(norm_image, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), mode='constant', constant_values=0) 
            pad_mask_norm = np.pad(norm_mask, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), mode='constant', constant_values=0)
        img_set_1[i] = pad_img_norm
        mask_set_1[i] = pad_mask_norm

    return img_set_1, mask_set_1

def get_image_array(image_path,mask_path,num_class = 2,image_as_gray = False,mask_as_gray = True):
    # store files
    image_name_arr = glob.glob(os.path.join(image_path,"*.tif"))
    mask_name_arr = glob.glob(os.path.join(mask_path,"*.tif"))
    # sort files
    image_name_arr.sort(key=lambda x: int(x.split('/')[4][:-4]))
    mask_name_arr.sort(key=lambda x: int(x.split('/')[4][:-4]))

    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = cv2.imread (item,cv2.IMREAD_COLOR)
        img = np.asarray(img, dtype=np.uint8)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(mask_name_arr[index],as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        image_arr.append(img)
        mask_arr.append(mask)

    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr, mask_arr

def get_test_image_array(image_path,image_as_gray = False):
    # store files
    image_name_arr = glob.glob(os.path.join(image_path,"*.tif"))
    image_name_arr.sort(key=lambda x: int(x.split('/')[4][:-4]))
    
    image_arr = []
    for index,item in enumerate(image_name_arr):
        img = cv2.imread (item,cv2.IMREAD_COLOR)
        img = np.asarray(img, dtype=np.uint8)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        image_arr.append(img)
    image_arr = np.array(image_arr)
    img_set_1 = np.ndarray((image_arr.shape[0], 128,128,3), dtype="float32")
    for i in range (image_arr.shape[0]):
        image = np.asarray(image_arr[i,:,:], dtype="float32")

        # image normalisation
        norm_image = cv2.normalize(image,None,alpha=0,beta=1,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # image padding at four sides
        padding_size = 14
        pad_img_norm = np.pad(norm_image, ((padding_size, padding_size), (padding_size, padding_size), (0, 0)), mode='constant', constant_values=0) 
        img_set_1[i] = pad_img_norm
        
    return img_set_1
    

def image_processing(img, mask, target_size = (128, 128),augmentation = True,padding = True, save = False):
    image_arr = np.empty((0,target_size[0],target_size[1],3),dtype=np.uint8)
    mask_arr = np.empty((0,target_size[0],target_size[1],1),dtype=np.uint8)
    for i in range (img.shape[0]):
        original_img,original_mask = adjustData(img[i],mask[i], data_augmentation = False, padding = padding, target_size = target_size, save = save)
        image_arr = np.vstack((image_arr,original_img))
        mask_arr = np.vstack((mask_arr, original_mask))
        new_imgs,new_masks = adjustData(img[i],mask[i], data_augmentation = augmentation, padding = padding, target_size = target_size, save = save)
        image_arr = np.vstack((image_arr,new_imgs))
        mask_arr = np.vstack((mask_arr, new_masks))

    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr

def saveResult(save_path,npyfile):
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        #img = img*255
        img = np.uint8(255*img)
        ####### crop is used only if padding is used
        img = img[14:114,14:114]

        #######
        #img[img>=100] = 255
        #img[img<100] = 0
        io.imsave(os.path.join(save_path,"%d_predict.tif"%(i+1)),img)


def elastic_transform(image, alpha, sigma, randx, randy, M, random_state=None):
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # set random displacement 
    dx = gaussian_filter((randx * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((randy * 2 - 1), sigma, mode="constant", cval=0) * alpha
    # apply elastic deformation
    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

def data_rotate_Crop(row_roi,label_roi,degree,size):
    shape = row_roi.size
    shape = shape[0]
    row_roi = row_roi.rotate(degree)
    label_roi = label_roi.rotate(degree)
    row_roi = row_roi.crop((size,size,shape-size,shape-size))
    label_roi = label_roi.crop((size,size,shape-size,shape-size))
    row_roi = row_roi.resize((shape,shape))
    label_roi = label_roi.resize((shape,shape))
    return row_roi,label_roi

def threshold(row_roi):
    row_roi = np.asarray(row_roi, dtype='int32')
    row_roi[row_roi > 128] = 255
    row_roi[row_roi <= 128] = 0
    row_roi = Image.fromarray(np.uint8(row_roi))
    return row_roi

def create_ED_dataset(img, mask, img_nums = 4,mode = 'orginal', save = False):
    count = 0
    shape = img.shape
    shape_size = shape[:2]
    random_state = np.random.RandomState(None)
    img = asarray(img)
    mask = asarray(mask)
    mask = np.concatenate((mask,)*3, axis=-1)

    # initilise array
    ED_img_set = np.ndarray((img_nums,shape[0],shape[1],3), dtype=np.uint8)
    ED_mask_set = np.ndarray((img_nums,shape[0],shape[1],1), dtype=np.uint8)

    while(count < img_nums):
        # alpha and sigma (parameter)
        alpha = shape[1] * 2.3
        sigma = shape[1] * 1.4
        # random displacement vector
        randx = random_state.rand(*shape)
        randy = random_state.rand(*shape)

        # Random affine (parameter)
        alpha_affine = shape[1]  * 0.07*( count + 1) #(parameter)
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)

        # for the raw image
        ED_img = elastic_transform(img, alpha, sigma, randx, randy, M)

        # for the mask image
        ED_mask_pre = elastic_transform(mask, alpha, sigma, randx, randy,M)
        # use threshold to convert to B&W image
        ED_mask = np.ndarray((shape[0],shape[1],1), dtype=np.uint8)
        ED_mask = ED_mask_pre[:,:,0]
        ED_mask[ED_mask > 128] = 255
        ED_mask[ED_mask <= 128] = 0
        ED_mask = ED_mask.reshape(ED_mask.shape+(1,))

        ED_img_set[count] = ED_img
        ED_mask_set[count] = ED_mask

        if save == True:

            uniq = 0
            row_img_store = 'data/membrane/test/test_image/%d.tif' % (uniq)
            label_img_store = 'data/membrane/test/test_label/%d.tif' % (uniq)
            while os.path.exists(row_img_store):
                row_img_store = 'data/membrane/test/test_image/%d.tif' % (uniq)
                label_img_store = 'data/membrane/test/test_label/%d.tif' % (uniq)
                uniq += 1
            Image.fromarray(ED_img).save(row_img_store)
            Image.fromarray(ED_mask[:,:,0]).save(label_img_store)

        count += 1

    return ED_img_set, ED_mask_set


def create_flip_set(img, mask,mode = 'orginal', save = False):
    count = 0
    rotate = [90, 180, 270,'a','b','c','d']
    # initilise array
    flip_img_set = np.ndarray((len(rotate),img.shape[0],img.shape[1],3), dtype=np.uint8)
    flip_mask_set = np.ndarray((len(rotate),img.shape[0],img.shape[1],1), dtype=np.uint8)

    row_img = Image.fromarray(np.uint8(img))
    label_img = Image.fromarray(np.uint8(mask[:,:,0]))

    for j in range(len(rotate)):
        if rotate[j] == 'a':
            row_roi = row_img.transpose(Image.FLIP_LEFT_RIGHT)
            label_roi = label_img.transpose(Image.FLIP_LEFT_RIGHT)
        elif rotate[j] == 'b':
            row_roi = row_img.rotate(90)
            label_roi = label_img.rotate(90)
            row_roi = row_roi.transpose(Image.FLIP_LEFT_RIGHT)
            label_roi = label_roi.transpose(Image.FLIP_LEFT_RIGHT)
        elif rotate[j] == 'c':
            row_roi = row_img.rotate(180)
            label_roi = label_img.rotate(180)
            row_roi = row_roi.transpose(Image.FLIP_LEFT_RIGHT)
            label_roi = label_roi.transpose(Image.FLIP_LEFT_RIGHT)
        elif rotate[j] == 'd':
            row_roi = row_img.rotate(270)
            label_roi = label_img.rotate(270)
            row_roi = row_roi.transpose(Image.FLIP_LEFT_RIGHT)
            label_roi = label_roi.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            row_roi = row_img.rotate(rotate[j])
            label_roi = label_img.rotate(rotate[j])

        # save and check
        if save == True:
            uniq = 0
            row_img_store = 'data/membrane/test/test_image/%d.tif' % (uniq)
            label_img_store = 'data/membrane/test/test_label/%d.tif' % (uniq)
            while os.path.exists(row_img_store):
                row_img_store = 'data/membrane/test/test_image/%d.tif' % (uniq)
                label_img_store = 'data/membrane/test/test_label/%d.tif' % (uniq)
                uniq += 1
            row_roi.save(row_img_store)
            label_roi.save(label_img_store)

        # store image array
        flip_img_set[count] = np.asarray(row_roi)
        label_roi = np.asarray(label_roi)
        flip_mask_set[count] = label_roi.reshape(label_roi.shape+(1,))
        count += 1
    return flip_img_set, flip_mask_set

def create_rotation_set(img, mask,mode = 'original', save = False):
    rotate = [30, 45, 135, 60, 120, 150]
    crop = [12, 12, 12, 12, 12, 12]

    # initilise array
    rotate_img_set = np.ndarray((len(rotate),img.shape[0],img.shape[1],3), dtype=np.uint8)
    rotate_mask_set = np.ndarray((len(rotate),img.shape[0],img.shape[1],1), dtype=np.uint8)

    for j in range(len(rotate)):
        row_roi, label_roi = data_rotate_Crop(Image.fromarray(np.uint8(img)),Image.fromarray(np.uint8(mask[:,:,0])),rotate[j],crop[j])

        # save and check
        if save == True:
            uniq = 0
            row_img_store = 'data/membrane/test/test_image/%d.tif' % (uniq)
            label_img_store = 'data/membrane/test/test_label/%d.tif' % (uniq)
            while os.path.exists(row_img_store):
                row_img_store = 'data/membrane/test/test_image/%d.tif' % (uniq)
                label_img_store = 'data/membrane/test/test_label/%d.tif' % (uniq)
                uniq += 1
            row_roi.save(row_img_store)
            label_roi.save(label_img_store)

        # store image array
        rotate_img_set[j] = np.asarray(row_roi)
        rotate_roi = np.asarray(label_roi)
        rotate_mask_set[j] = rotate_roi.reshape(rotate_roi.shape+(1,))

    return rotate_img_set, rotate_mask_set
            
def crop_image_set(img, mask, mode = 'original', save = False):
    g_count = 0
    size = 70   #cropping size
    start_x = [0,0,30,30,15]
    start_y = [0,30,0,30,15]

    # initilise array
    crop_img_set = np.ndarray((len(start_x),img.shape[0],img.shape[1],3), dtype=np.uint8)
    crop_mask_set = np.ndarray((len(start_x),img.shape[0],img.shape[1],1), dtype=np.uint8)

    row_img = Image.fromarray(np.uint8(img))
    label_img = Image.fromarray(np.uint8(mask[:,:,0]))

    for j in range(len(start_x)):
        row_roi = row_img.crop((start_x[j],start_y[j],start_x[j]+size,start_y[j]+size))
        label_roi = label_img.crop((start_x[j],start_y[j],start_x[j]+size,start_y[j]+size))
        row_roi = row_roi.resize((100,100))
        label_roi = label_roi.resize((100,100))
        label_roi= threshold(label_roi)

        # save and check
        if save == True:
            uniq = 0
            row_img_store = 'data/membrane/test/test_image/%d.tif' % (uniq)
            label_img_store = 'data/membrane/test/test_label/%d.tif' % (uniq)
            while os.path.exists(row_img_store):
                row_img_store = 'data/membrane/test/test_image/%d.tif' % (uniq)
                label_img_store = 'data/membrane/test/test_label/%d.tif' % (uniq)
                uniq += 1
            row_roi.save(row_img_store)
            label_roi.save(label_img_store)
        
        # store image array
        crop_img_set [j] = np.asarray(row_roi)
        crop_roi = np.asarray(label_roi)
        crop_mask_set[j] = crop_roi.reshape(crop_roi.shape+(1,))

        g_count += 1
    return crop_img_set, crop_mask_set


