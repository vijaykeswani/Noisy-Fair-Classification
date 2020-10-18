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
from utils import *

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
    
