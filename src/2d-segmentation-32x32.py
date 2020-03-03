import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import seaborn as sns
plt.style.use("ggplot")


from tqdm import tqdm, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

import keras
from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import (Input, BatchNormalization, Activation, Dense, Dropout, Flatten, 
                          Conv3D, Conv3DTranspose, MaxPooling3D)
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from numpy import genfromtxt

from unet_2d import get_unet



# Set some parameters
resize_width = 128
resize_height = 128

im_depth = 32
im_width = 32
im_height = 32
border = 5



ids = sorted(next(os.walk("../data/image"))[2]) # list of names all images in the given path
print("No. of images = ", len(ids))
print(ids)

X = np.zeros((len(ids), resize_width, resize_height, 1), dtype=np.float32)
y = np.zeros((len(ids), resize_width, resize_height, 1), dtype=np.float32)

for n, id_ in tqdm(enumerate(ids), total=len(ids)):
    # Load images
    img = np.loadtxt(open("../data/image/"+id_, "rb"), delimiter= ',')
    x_img = img.T
    x_img = resize(x_img, (128, 128, 1), mode = 'constant', preserve_range = True)
    # Save images
    X[n] = x_img

for n, id_ in tqdm(enumerate(ids), total=len(ids)):
    # Load mask
    id_ = "msk_"+id_[-7:]
    mask = np.loadtxt(open("../data/mask/"+id_, "rb"), delimiter= ',')
    mask = mask.T
    mask = resize(mask, (128, 128, 1), mode = 'constant', preserve_range = True)    
    # Save mask
    y[n] = mask


#cut X y data into 3d cubes of dim = im_depth * im_depth *im_height
X_3d = np.zeros((676//im_depth*(128//im_depth)*(128//im_height), im_depth, im_height, im_width, 1), dtype=np.float32)
y_3d = np.zeros((676//im_depth*(128//im_depth)*(128//im_height), im_depth, im_height, im_width, 1), dtype=np.float32)


for i in range(676//im_depth):
    for j in range(128//im_width):
        for k in range(128//im_height):
            indx =  i*((128//im_width)*(128//im_height))+j*(128//im_height)+k
            print('3d cube index:', i, j, k, indx)
            X_3d[indx]=X[i*im_depth:(i+1)*im_depth, j*im_width:(j+1)*im_width, k*im_height:(k+1)*im_height]
            y_3d[indx]=y[i*im_depth:(i+1)*im_depth, j*im_width:(j+1)*im_width, k*im_height:(k+1)*im_height]

#For fair comparison with 3d segmentation
# using the same set of input data

X_2d = X_3d.reshape(336*32, 32, 32, 1)
y_2d = y_3d.reshape(336*32, 32, 32, 1) 


# Split train and valid
X_train, X_valid, y_train, y_valid = train_test_split(X_2d, y_2d, test_size=0.2, random_state=23)

input_img = Input((im_height, im_width, 1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('2d-salt_32x32.h5', verbose=1, save_best_only=True, save_weights_only=True)
]

results = model.fit(X_train, y_train, batch_size=256, epochs=50, callbacks=callbacks,\
                    validation_data=(X_valid, y_valid))


plt.figure(figsize=(8, 8))
plt.title("Learning curve")
plt.plot(results.history["loss"], label="loss")
plt.plot(results.history["val_loss"], label="val_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
plt.xlabel("Epochs")
plt.ylabel("log_loss")
plt.legend()