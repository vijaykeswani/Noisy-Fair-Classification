import numpy as np

import sys
# sys.path.append("/home/ec2-user/experiment/AIF360/")

# import aif360.datasets.noisy_dataset as noisy
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



import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression

attributes = ['race', 'sex']
resultsByAttribute = {}

attr = sys.argv[1]
print ("Attribute", attr)
reps = 10
results = {"denoised-sr": {}, "denoised-fpr": {}, "eo": {}, "unconstrained": {}, "undenoised-sr": {}, "undenoised-fpr": {}, "lamy": {}}
lam = 0.1
C = 0
thresh = 0.1

eta0 = float(sys.argv[2])
eta1 = float(sys.argv[3])
tau = float(sys.argv[4])

def step(rep):
    print ("Started repetition", rep)
    results = {}
    dataset = load_preproc_data_compas()

    protected_name = attr
    sensible_name = attr

    privileged_groups = [{protected_name: 1}]
    unprivileged_groups = [{protected_name: 0}]

    dataset_train, dataset_test = dataset.split([0.7], shuffle=True)
    train_labels = [int(lab[0]) for lab in dataset_train.labels]
    test_labels = [int(lab[0]) for lab in dataset_test.labels]
    index, noisyfea = flipping(dataset_train.feature_names, dataset_train.features, train_labels, protected_name, eta0, eta1)

    dataset_noisy = np.copy(dataset_train.features)
    dataset_noisy[:,index] = noisyfea

    train_labels = [int(lab[0]) for lab in dataset_train.labels]

    clf = LogisticRegression(random_state=0).fit(dataset_noisy, train_labels)
    y_true_train = train_labels
    y_pred_train = clf.predict(dataset_noisy)
    group_train = dataset_noisy[:,index]

    index, test_noisyfea = flipping(dataset_test.feature_names, dataset_test.features, test_labels, protected_name, eta0, eta1)
    dataset_noisy_test = np.copy(dataset_test.features)
    dataset_noisy_test[:,index] = test_noisyfea

    ### Baseline - Unconstrained
    y_pred_test = clf.predict(dataset_noisy_test)
    results["unconstrained"] = getStats(y_pred_test, test_labels, dataset_test.features[:, index])    
    print ("Unconstrained:", results["unconstrained"])


    ### Baseline - LR with unmodified statistical rate constraints
    undenoised_theta = denoisedfair.undenoised(dataset_noisy, dataset_train.labels, index, C, tau)
    acc, sr, disp, fpr = testing(dataset_noisy_test, dataset_test.features[:,index], dataset_test.labels, index, undenoised_theta)
    results["undenoised-sr"] =  {"acc":acc, "sr":sr, "fpr":fpr}
    print ("Undenoised-sr:", results["undenoised-sr"], "\n")

    ### Baseline - LR with unmodified false positive rate constraints
    undenoised_theta = denoisedfair.undenoised_eo(dataset_noisy, dataset_train.labels, index, C, tau)
    acc, sr, disp, fpr = testing(dataset_noisy_test, dataset_test.features[:,index], dataset_test.labels, index, undenoised_theta)
    results["undenoised-fpr"] = {"acc":acc, "sr":sr, "fpr":fpr}
    print ("Undenoised-fpr:", results["undenoised-fpr"], "\n")

    ## LR with modified statistical rate constraints
    denoised_theta = denoisedfair.denoised(dataset_noisy, dataset_train.labels, index, C, tau, lam, eta0, eta1)
    acc, sr, disp, fpr = testing(dataset_noisy_test, dataset_test.features[:,index], dataset_test.labels, index, denoised_theta)

    results["denoised-sr"] = {"acc":acc, "sr":sr, "fpr":fpr}
    print("Denoised-SR:", results["denoised-sr"], "\n")

    ## LR with modified false positive rate constraints
    denoised_theta = denoisedfair.denoised_eo(dataset_noisy, dataset_train.labels, index, C, tau, lam, eta0, eta1)
    acc, sr, disp, fpr = testing(dataset_noisy_test, dataset_test.features[:,index], dataset_test.labels, index, denoised_theta)
    results["denoised-fpr"] = {"acc":acc, "sr":sr, "fpr":fpr}
    print("Denoised-FPR:", results["denoised-fpr"], "\n")

    
    dataset_all = {"dataset_train": dataset_train, "dataset_train_noisy": dataset_noisy, "dataset_test": dataset_test, "dataset_test_noisy": dataset_noisy_test}
    print ("Ended repetition", rep)
    
    ret = (results, dataset_all)
    np.save("results/results_compas_noisy_eta_" + attr + "_rep" + str(rep) + ".npy", ret)

    return results, dataset_all


from multiprocessing import Pool
reps = int(sys.argv[5])
with Pool(5) as p:
    r = list(tqdm(p.imap(step, range(reps)), total=reps))
    
