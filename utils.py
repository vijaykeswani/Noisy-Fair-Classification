import numpy as np

import sys
import algorithms as denoisedfair
from aif360.datasets import BinaryLabelDataset
from aif360.datasets import AdultDataset, CompasDataset

from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas

import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
import random
from copy import deepcopy
random.seed()
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression


# insert flipping noises in the selected feature
def flipping(feature_names, features, labels, name, eta0, eta1):
    index = feature_names.index(name)
    N = features.shape[0]
    noisyfea = deepcopy(features[:,index])
    count = 0
    for i in range(N):
        seed = random.random()
        if int(features[i][index]) == 1:
            if seed < eta1:
                noisyfea[i] = 1 - noisyfea[i]
                count += 1
        elif int(features[i][index]) == 0:
            if seed < eta0:
                noisyfea[i] = 1 - noisyfea[i]
                count += 1
    print('Count_flipping:', count)
    return index, noisyfea

def testing(features, groups, labels, index, theta):
    N = features.shape[0]
    d = features.shape[1]
    N1 = sum(groups)
    N0 = N - N1
    NTrue = 0
    N0True = 0
    N1True = 0
    
    Ny = len(labels) - sum(labels)
    Ny1 = sum([1 if groups[i] == 1 and labels[i] == 0 else 0 for i in range(len(labels))])
    Ny0 = sum([1 if groups[i] == 0 and labels[i] == 0 else 0 for i in range(len(labels))])
    Ny0True, Ny1True = 0, 0
    
    X = np.zeros([N, d + 1])
    X[:, 0:d] = features
    X[:, d] = [1 for i in range(N)]
    X = np.delete(X, index, 1)
    for i in range(N):
        predict = 1 / (1 + np.exp(-np.dot(theta, X[i])))
        if predict >= 0.5:
            predict = 1
        else:
            predict = 0
        if labels[i] == predict:
            NTrue += 1
        if predict == 1:
            if int(groups[i]) == 1:
                N1True += 1
            else:
                N0True += 1
                
            if labels[i] == 0:
                if int(groups[i]) == 1:
                    Ny1True += 1
                else:
                    Ny0True += 1
                
        
    acc = NTrue / N
    sr0 = N0True / N0
    sr1 = N1True / N1
    if (sr0 == 0) & (sr1 == 0):
        sr = 1
    elif (sr0 == 0) or (sr1 == 0):
        sr = 0
    else:
        sr = min(sr0 / sr1, sr1 / sr0)
        
    fpr0 = Ny0True/Ny0
    fpr1 = Ny1True/Ny1
    if (fpr0 == 0) & (fpr1 == 0):
        fpr = 1
    elif (fpr0 == 0) or (fpr1 == 0):
        fpr = 0
    else:
        fpr = min(fpr0 / fpr1, fpr1 / fpr0)
    
    
    
    return acc, sr, np.abs(sr0 - sr1), fpr

def getStats(predictions, labels, groups):
    N = len(labels)
    N1 = sum(groups)
    N0 = N - N1

    Ny = len(labels) - sum(labels)
    Ny1 = sum([1 if groups[i] == 1 and labels[i] == 0 else 0 for i in range(len(labels))])
    Ny0 = sum([1 if groups[i] == 0 and labels[i] == 0 else 0 for i in range(len(labels))])
    Ny0True, Ny1True, NTrue, N0True, N1True = 0, 0, 0, 0, 0
    
    for i in range(N):
        predict = predictions[i]
        if labels[i] == predict:
            NTrue += 1
        if predict == 1:
            if labels[i] == 0:
                if int(groups[i]) == 1:
                    Ny1True += 1
                else:
                    Ny0True += 1

            if int(groups[i]) == 1:
                N1True += 1
            else:
                N0True += 1
                
                
    acc = NTrue/N
    fpr0 = Ny0True/Ny0
    fpr1 = Ny1True/Ny1
    if (fpr0 == 0) & (fpr1 == 0):
        fpr = 1
    elif (fpr0 == 0) or (fpr1 == 0):
        fpr = 0
    else:
        fpr = min(fpr0 / fpr1, fpr1 / fpr0)
    
    sr0 = N0True / N0
    sr1 = N1True / N1
    if (sr0 == 0) & (sr1 == 0):
        sr = 1
    elif (sr0 == 0) or (sr1 == 0):
        sr = 0
    else:
        sr = min(sr0 / sr1, sr1 / sr0)
        
    
    
    return {"acc":acc, "sr":sr, "fpr":fpr}


