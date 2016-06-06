from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

json_string = open("model/structure.txt", "r").read()
model = model_from_json(json_string)
model.load_weights("weights.txt")
"""
model.layers.pop()
model.layers[-1].outbound_nodes = []
model.outputs = [model.layers[-1].output]
"""
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

import os
import re
import numpy as np
from PIL import Image
def img_to_array(img, dim_ordering='th'):
    x = np.asarray(img, dtype='float32')
    if dim_ordering == 'th':
        x = x.transpose(2, 0, 1)
    return x
def load_img(path, grayscale=False):
    img = Image.open(path).resize((64*2, 48*2))
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    return img
def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return sorted (
            [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and re.match('([\w]+\.(?:' + ext + '))', f)]
           )

def define(category):
    label = "???"
    if category == 0:
        label = "regular driving"
    if category == 1:
        label = "right text"
    if category == 2:
        label = "right call"
    if category == 3:
        label = "left text"
    if category == 4:
        label = "left call"
    if category == 5:
        label = "radio"
    if category == 6:
        label = "drink"
    if category == 7:
        label = "back"
    if category == 8:
        label = "hair"
    if category == 9:
        label = "talk"
    return label

all_files = list_pictures("data/test")
fout = open('model/results.txt', 'w')
for start in range(0, len(all_files), 10000):
    F = all_files[start:min(len(all_files),start+10000)]
    X = np.array([img_to_array(load_img(file)) for file in F])
    X = X.astype('float32')
    X /= 255.0
    y = model.predict_proba(X, batch_size=32, verbose=1)
    for i in range(len(F)):
        c1 = np.argmax(y[i])
        l1 = define(c1)
        fout.write(str(F[i]) + " " + str(c1) + " " + l1 + "\n")
fout.close()
