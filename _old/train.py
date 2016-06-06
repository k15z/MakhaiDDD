"""
c0 - regular driving
c1 - right hand texting
c2 - right hand calling
c3 - left hand texting
c4 - left hand calling
c5 - radio
c6 - drink
c7 - reach back
c8 - brush hair
c9 - talk to passenger
"""

import pandas
import random
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import os
import re
import math
import numpy as np
from PIL import Image

print "Loading files..."
split = {}
testData = []
trainData = []
df = pandas.read_csv("data/driver_imgs_list.csv")
for i in range(df.shape[0]):
    if df['subject'][i] not in split:
        if random.random() < 0.4:
            split[df['subject'][i]] = "test"
        else:
            split[df['subject'][i]] = "train"
    if split[df['subject'][i]] == "test":
        testData += [(int(df['classname'][i][1:]), "data/train/"+df['classname'][i]+"/"+df['img'][i])]
    if split[df['subject'][i]] == "train":
        trainData += [(int(df['classname'][i][1:]), "data/train/"+df['classname'][i]+"/"+df['img'][i])]

print "Loading model..."
batch_size = 32
nb_classes = 10
nb_epoch = 4
img_rows, img_cols = 48*2, 64*2
img_channels = 3
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
structure = open("model/structure.txt", "w")
structure.write(model.to_json())
structure.close()

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
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f)) and re.match('([\w]+\.(?:' + ext + '))', f)]

print "Loading images..."
X_train = np.concatenate([[img_to_array(load_img(image)) for label, image in trainData]],axis=0)
X_test = np.concatenate([[img_to_array(load_img(image)) for label, image in testData]],axis=0)
y_train = np.array([label for label, image in trainData]).transpose()
y_test = np.array([label for label, image in testData]).transpose()

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

model_checkpoint = ModelCheckpoint("model/weights.{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}.txt")
model.fit((X_train, Y_train),
          batch_size=32,
          samples_per_epoch=len(X_train),
          nb_epoch = nb_epoch,
          validation_data=(X_test, Y_test),
          callbacks=[model_checkpoint])

# train

y_out = model.predict_proba(X_test, batch_size=128, verbose=1)

# Part 1
print("Analyzing results...")
score = 0
logloss = 0.0
table = [[0]*10 for y in range(10)]
for i in range(len(y_out)):
    actual = y_test[i]
    predict = np.argmax(y_out[i])
    table[actual][predict] += 1
    if actual == predict:
        score+=1
        logloss += math.log(1-0.01)
    else:
        logloss += math.log(0.01/9)
print(logloss/len(y_out))
output = "c0\tc1\tc2\tc3\tc4\tc5\tc6\tc7\tc8\tc9\n"
for actual in range(10):
    for predict in range(10):
        output += str(table[actual][predict]) + "\t"
    output += "\n"
print(output)

# Part 2
print("Analyzing results...")
table = [[0]*10 for y in range(10)]
for i in range(len(y_out)):
    trial = 0
    while np.argmax(y_out[i]) != y_test[i]:
        trial += 1
        y_out[i][np.argmax(y_out[i])] = -1e10
    predict = np.argmax(y_out[i])
    table[trial][predict] += 1
output = "c0\tc1\tc2\tc3\tc4\tc5\tc6\tc7\tc8\tc9\n"
for actual in range(10):
    for predict in range(10):
        output += str(table[actual][predict]) + "\t"
    output += "\n"
print(output)
