## Python implementation of STRIP

import os
import sys
import random
import h5py
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model



## loading the data from the .h5 files
def load_data(file_path):
    data = h5py.File(file_path, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))
    y_data = to_categorical(y_data, num_classes=1283)
    return x_data, y_data



## converting the range of image values from (0, 255) to (0, 1)
def preprocess(x):
    return x / 255.0


## sorting the clean samples' dataset
def sort_samples(xval, yval, num_classes):    
    index = np.zeros((1283, 9)).astype('int')
    cls_cnt = np.zeros(1283).astype('int')
    yval = np.argmax(yval, axis=1)
    for i in range(xval.shape[0]):

        ## index[c, x] is the index of the x-th image in class c under xval
        index[yval[i], cls_cnt[yval[i]]] = i
        cls_cnt[yval[i]] += 1
    

    D = []
    for cls in range(num_classes):
        cls_samples = []
        for i in range(9):
            cls_samples.append(xval[index[cls, i]])
        ## appending a new list with the same nine classes of images
        D.append(cls_samples)
    D = np.array(D)
    
    return D


## displaying images in sorted dataset D under the list cls_list
def show(D, cls_list):
    n = len(cls_list)
    plt.figure(figsize=(2*n,n))
    for j, cls in enumerate(cls_list):
        for i in range(9):
            plt.subplot(9, 2*n, j*2*n+i+1)
            plt.imshow(D[cls, i])


## blending two imgages
def perturbe(b, t, alpha):
    assert alpha < 1 and alpha > 0, "r must satisfy 0 < r < 1"
    return b*alpha+t*(1-alpha)


## generating perturbed inputs for the BadNet 
def perturbation_step(x, D, alpha):
    assert alpha < 1 and alpha > 0, "r must satisfy 0 < r < 1"
    N = len(D)
    perturbed_x = []
    for i in range(N):
        a = random.randint(0, N-1)
        b = random.randint(0, 8)
        perturbed_x.append(x)
        perturbed_x[i] = perturbe(perturbed_x[i], D[a, b], alpha)
    perturbed_x = np.array(perturbed_x)
    return perturbed_x

## entropy
def entroy(logit):
    prob = logit / np.sum(logit)
    sum = 0
    for p in prob:
        if p == 0: item = 0
        else: item = - p * np.log2(p)
        sum += item
    return sum


# judging whether a single input x is a trojaned input. If yes, return the prediction and H
def detect_trojan(model, x, D, boundary):
    N = len(D)
    H = 0
    perturbed_x = perturbation_step(x, D, 0.5)
    logits = model.predict(perturbed_x)
    
    for logit in logits:
        H_i = entroy(logit)
        H += H_i
    H /= N

    if H <= boundary:
        pred = N
    else:
        logit = model.predict(np.expand_dims(x, axis=0))
        pred = np.argmax(logit, axis=1)
    
    return pred, H


## perturbation step batch
def perturbation_step_batch(x, D, alpha):
    nsamp = x.shape[0]
    N = len(D)
    replicas = np.expand_dims(x, 1).repeat(9, axis=1)
    i = np.random.randint(0, N-1, size=[nsamp, 9])
    cls = np.random.randint(0,8,size=[nsamp, 9])
    random_test_samples = D[i, cls]
    perturbed_inputs = perturbe(replicas, random_test_samples, alpha)
    return perturbed_inputs


## entropy batch
def entroy_batch(logits):
    H_n = []
    for logit in logits:
        prob = logit / np.sum(logit)
        sum = 0
        for p in prob:
            if p == 0: item = 0
            else: item = - p * np.log2(p)
            sum += item
        H_n.append(sum)
    H_n = np.array(H_n)
    return H_n


## detecting the trojan
def detect_trojan_batch(model, x, D, boundary):
    N = len(D)
    pred = []

    logits = model.predict(x)

    perturbed_inputs = perturbation_step_batch(x, D, 0.5)
    
    H_list = []
    for i in tqdm(range(N)):
        perturbed_input = perturbed_inputs[i]
        logits = model.predict(perturbed_input)
        H_n = entroy_batch(logits)
        H = np.sum(H_n)
        H_list.append(H)
        if (H < boundary):
            pred.append(N)
        else:
            pred.append(np.argmax(logits[i]))
    H_list = np.array(H_list)


def G(model_path, data_path):
    num_class = 1283

    valid_data_path = 'data/clean_validation_data.h5'

    xval, yval = load_data(valid_data_path)
    xval = preprocess(xval)
    D = sort_samples(xval, yval, num_class)

    badnet = load_model(model_path)
    x, y = load_data(data_path)
    x = preprocess(x)

    pred = detect_trojan_batch(badnet, x, D, 0.5)

    return pred


def eval(model_path, img_path):
    
    xval, yval = load_data('data/clean_validation_data.h5')
    xval = preprocess(xval)
    D = sort_samples(xval, yval, 1283)

    badnet = load_model(model_path)
    x = plt.imread(img_path)[:,:,0:3]
    
    pred = detect_trojan(badnet, x, D, 0.5)

    print(pred[0])