import numpy as np
import os, glob, math, cv2, time
from joblib import Parallel, delayed

import pandas
import random

def process(data):
    img_label, img_file = data
    img = cv2.imread(img_file)
    img = cv2.resize(img, (128, 96)).transpose((2,0,1)).astype('float32') / 255.0
    return (img_label, img)

def load_dataset():
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
    testData = Parallel(n_jobs=-1,verbose=1)(delayed(process)(data) for data in testData)
    trainData = Parallel(n_jobs=-1,verbose=1)(delayed(process)(data) for data in trainData)
    random.shuffle(testData)
    random.shuffle(trainData)
    return (
        (
            [image for label, image in trainData],
            [label for label, image in trainData]
        ),
        (
            [image for label, image in testData],
            [label for label, image in testData]
        )
    )

def load_testset():
    files = glob.glob(os.path.join('data/test', '*.jpg'))
    testData = Parallel(n_jobs=nprocs)(delayed(process)((im_file, im_file)) for im_file in files)
    X_test, X_test_id = zip(*results)
    return (
        [image for label, image in testData],
        [label for label, image in testData]
    )
