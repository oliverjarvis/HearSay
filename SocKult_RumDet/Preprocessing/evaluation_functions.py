#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code contains several versions of evaluation functions
"""
import numpy as np
from model import LSTM_model_veracity

import os
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from keras.utils.np_utils import to_categorical
from branch2treelabels import branch2treelabels_test
import pickle
from copy import deepcopy
#%%

def evaluation_function_veracity_branchLSTM(params):

    """x_train = np.load(os.path.join(path, 'train/train_array.npy'))
    y_train = np.load(os.path.join(path, 'train/labels.npy'))
    x_dev = np.load(os.path.join(path, 'dev/train_array.npy'))
    y_dev = np.load(os.path.join(path, 'dev/labels.npy'))
    x_test = np.load(os.path.join(path, 'test/train_array.npy'))
    ids_test = np.load(os.path.join(path, 'test/ids.npy'))
    # join dev and train
    x_dev = pad_sequences(x_dev, maxlen=len(x_train[0]), dtype='float32',
                          padding='post', truncating='post', value=0.)
    x_train = np.concatenate((x_train, x_dev), axis=0)
    y_train = np.concatenate((y_train, y_dev), axis=0)
    y_train = to_categorical(y_train, num_classes=None)"""
    
    
    x_train_embeddings = np.load("saved_dataSocKult_RumDet/train/embeddings_array.npy")
    x_train_metafeatures = np.load("saved_dataSocKult_RumDet/train/metafeatures_array.npy")

    # Loading the veracity of the tweets True = 0, False = 1, Unverified = 2
    y_train = np.load("saved_dataSocKult_RumDet/train/labels.npy")
    ## One-hot encoding veracity
    y_train_cat = []
    for i in range(len(y_train)):
        y_train_cat.append(to_categorical(y_train[i], num_classes=3))
    y_train_cat = np.asarray(y_train_cat)

    # Loading the dev features (even though we still call it test, might change)
    x_test_embeddings = np.load("saved_dataSocKult_RumDet/dev/embeddings_array.npy")
    x_test_metafeatures = np.load("saved_dataSocKult_RumDet/dev/metafeatures_array.npy")

    # Loading the veracity of the dev set
    y_test = np.load("saved_dataSocKult_RumDet/dev/labels.npy")
    
    # Loading the ids of the dev set
    ids_test = np.load("saved_dataSocKult_RumDet/dev/tweet_ids.npy")


    # Getting predictions and confidence of the model
    y_pred, confidence = LSTM_model_veracity(x_train_embeddings, x_train_metafeatures, y_train_cat,
                                           x_test_embeddings, x_test_metafeatures, params)


    
    #Getting the predictions of trees and the branches
    trees, tree_prediction, tree_label, _ = branch2treelabels(ids_test, 
                                                              y_test,
                                                              y_pred,
                                                              confidence)
    
        
    return trees, tree_prediction, tree_confidence
