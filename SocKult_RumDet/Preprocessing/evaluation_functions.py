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
    #TODO: 
    
    """path = "preprocessing/saved_dataRumEval2019"
    x_train = np.load(os.path.join(path, 'train/train_array.npy'))
    y_train = np.load(os.path.join(path, 'train/fold_stance_labels.npy'))

    print (x_train.shape)

#    ids_train = np.load(os.path.join(path, 'train/tweet_ids.npy'))
    x_dev = np.load(os.path.join(path, 'dev/train_array.npy'))
    y_dev = np.load(os.path.join(path, 'dev/fold_stance_labels.npy'))
#    ids_dev = np.load(os.path.join(path, 'dev/tweet_ids.npy'))
    x_test = np.load(os.path.join(path, 'test/train_array.npy'))
#    y_test = np.load(os.path.join(path, 'test/fold_stance_labels.npy'))
    ids_test = np.load(os.path.join(path, 'test/tweet_ids.npy'))
    # join dev and train
    x_dev = pad_sequences(x_dev, maxlen=len(x_train[0]), dtype='float32',
                          padding='post', truncating='post', value=0.)
    y_dev = pad_sequences(y_dev, maxlen=len(y_train[0]), dtype='float32',
                          padding='post', truncating='post', value=0.)
    
    x_train = np.concatenate((x_train, x_dev), axis=0)
    y_train = np.concatenate((y_train, y_dev), axis=0)
    y_train_cat = []
    for i in range(len(y_train)):
        y_train_cat.append(to_categorical(y_train[i], num_classes=4))
    y_train_cat = np.asarray(y_train_cat)
    y_pred, _ = LSTM_model_stance(x_train, y_train_cat,
                                           x_test, params,eval=True )"""


    # Loading training features  
    x_train_embeddings = np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\saved_dataSocKult_RumDet\\train\\embeddings_array.npy")
    x_train_metafeatures = np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\saved_dataSocKult_RumDet\\train\\metafeatures_array.npy")

    # Loading the veracity of the tweets True = 0, False = 1, Unverified = 2
    y_train =  np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\saved_dataSocKult_RumDet\\train\\labels.npy")
    ## One-hot encoding veracity
    y_train_cat = []
    for i in range(len(y_train)):
        y_train_cat.append(to_categorical(y_train[i], num_classes=3))
    y_train_cat = np.asarray(y_train_cat)

    # Loading the dev features (even though we still call it test, might change)
    x_test_embeddings =  np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\saved_dataSocKult_RumDet\\dev\\embeddings_array.npy")
    x_test_metafeatures = np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\saved_dataSocKult_RumDet\\dev\\metafeatures_array.npy")

    x_test_embeddings = pad_sequences(x_test_embeddings, maxlen=25,
                                         dtype='float32',
                                         padding='post',
                                         truncating='post', value=0)

    x_test_metafeatures = pad_sequences(x_test_metafeatures, maxlen=25,
                                         dtype='float32',
                                         padding='post',
                                         truncating='post', value=0)
    # Loading the veracity of the dev set
    y_test =  np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\saved_dataSocKult_RumDet\\dev\\labels.npy")
    
    # Loading the ids of the dev set
    ids_test =  np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\saved_dataSocKult_RumDet\\dev\\tweet_ids.npy", allow_pickle=True)


    # Getting predictions and confidence of the model
    y_pred, confidence = LSTM_model_veracity(x_train_embeddings, x_train_metafeatures, y_train_cat,
                                           x_test_embeddings, x_test_metafeatures, params)


    
    #Getting the predictions of trees and the branches
    trees, tree_prediction, tree_label, _ = branch2treelabels(ids_test, 
                                                              y_test,
                                                              y_pred,
                                                              confidence)
    
        
    return trees, tree_prediction, tree_confidence


# %%
