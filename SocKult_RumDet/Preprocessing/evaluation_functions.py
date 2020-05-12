#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code contains several versions of evaluation functions
"""
import numpy as np
#from model import LSTM_model_veracity
from model import LSTM_model_veracity
import os
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from keras.utils.np_utils import to_categorical
from branch2treelabels import branch2treelabels
import pickle
from copy import deepcopy
from pathlib import Path

data_dir = Path("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\sacred-twitter-data-w-test\\")
embeddings_name = "embeddings_array.npy"
metafeatures_name = "feature_array.npy"
labels_name = "labels.npy"
ids_name = "tweet_ids.npy"

# the sad consequence of shitty preprocessing
metafeatures = {
    'favorite_count' : (0, 1),
    'retweet_count' : (1, 2), 
    'user.verified' : (2, 3),
    'user.followers_count' : (3, 4),
    'user.listed_count' : (4, 5),
    'user.friends_count' : (5, 6),
    'user.favorites_count' : (6, 7),
    'stance' : (7, 11),
    'cosine_similarity' : (11, 12)
}

metafeaturegroups = {
    "cosine_similarity" : ['cosine_similarity'],
    "stance" : ['stance'],
    "social_interest" : ['favorite_count', 'retweet_count'],
    "user_information" : ['user.verified', 'user.followers_count', 'user.listed_count', 'user.friends_count', 'user.favorites_count']
}

def load_data_from_dir(dir, num_classes=3, with_ids=False):
    embeddings = np.load( data_dir / dir / embeddings_name )
    metafeatures = np.load( data_dir / dir / metafeatures_name )
    labels = np.load( data_dir / dir / labels_name)
    if num_classes > 0:
        labels = to_categorical(labels, num_classes=num_classes)
    if with_ids:
        ids = np.load( data_dir / dir / ids_name )
        return (embeddings, metafeatures, labels, ids)
    return (embeddings, metafeatures, labels)

def split_features(data, metagroups):
    new_metafeature = np.empty((data.shape[0],data.shape[1],0))
    for m in metagroups:
        features_fetch = metafeaturegroups[m]
        for feat in features_fetch:
            x1, x2 = metafeatures[feat]
            new_metafeature = np.concatenate((new_metafeature, data[:,:,x1:x2]), axis=2)
    return new_metafeature

#%%
def evaluation_function_veracity_branchLSTM(params, metac):
    
    x_train_embeddings, x_train_metafeatures, y_train = load_data_from_dir("train", num_classes=3)
    x_dev_embeddings, x_dev_metafeatures, y_dev = load_data_from_dir("dev", num_classes=3)
    x_test_embeddings, x_test_metafeatures, y_test, ids_test = load_data_from_dir("test", num_classes=0, with_ids=True)

    x_train_metafeatures = split_features(x_train_metafeatures, metac)
    x_dev_metafeatures = split_features(x_dev_metafeatures, metac)
    x_test_metafeatures = split_features(x_test_metafeatures, metac)
    
    x_train_embeddings = np.concatenate((x_train_embeddings, x_dev_embeddings), axis = 0)
    x_train_metafeatures = np.concatenate((x_train_metafeatures, x_dev_metafeatures), axis = 0)
    y_train = np.concatenate((y_train, y_dev), axis=0)

    # Getting predictions and confidence of the model
    y_pred, confidence = LSTM_model_veracity(x_train_embeddings, x_train_metafeatures, y_train,
                                        x_test_embeddings, x_test_metafeatures, params)
    
    #Getting the predictions of trees and the branches
    trees, tree_prediction, tree_label, tree_confidence = branch2treelabels(ids_test, 
                                                            y_test,
                                                            y_pred,
                                                            confidence)
    
    mactest_F = f1_score(tree_label, tree_prediction, average='macro')
            
    return trees, tree_prediction, tree_label, tree_confidence, mactest_F



"""

    x_test_stance = x_test_metafeatures[:,:,-5:-1]
    x_test_metafeatures = np.concatenate((x_test_metafeatures[:,:,:-5], x_test_metafeatures[:,:,-1:]), axis=2)
    #x_test_metafeatures = np.concatenate((x_test_stance, x_test_metafeatures), axis=
    
    # Testing with and without stance
    #x_test_metafeatures = np.concatenate((x_test_stance, x_test_metafeatures), axis=2)
  

    #y_train = (to_categorical(y_train, num_classes=3))
    #print(y_train)
    # Loading training features  
    x_train_embeddings = np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\sacred-twitter-data-w-test\\train\\embeddings_array.npy")
    x_train_metafeatures = np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\sacred-twitter-data-w-test\\train\\feature_array.npy")

    # Loading the veracity of the tweets True = 0, False = 1, Unverified = 2
    y_train =  np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\sacred-twitter-data-w-test\\train\\labels.npy")

    x_train_stance = x_train_metafeatures[:,:,-5:-1]
    x_train_metafeatures = np.concatenate((x_train_metafeatures[:,:,:-5], x_train_metafeatures[:,:,-1:]), axis=2)
    #x_train_metafeatures = np.concatenate((x_train_stance, x_train_metafeatures), axis=2)

    # Loading the dev features (even though we still call it test, might change)
    x_dev_embeddings =  np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\sacred-twitter-data-w-test\\dev\\embeddings_array.npy")
    x_dev_metafeatures = np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\sacred-twitter-data-w-test\\dev\\feature_array.npy")

    # Loading the veracity the tweets True = 0, False = 1, Unverified = 2
    y_dev =  np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\sacred-twitter-data-w-test\\dev\\labels.npy")

    x_dev_stance = x_dev_metafeatures[:,:,-5:-1]
    x_dev_metafeatures = np.concatenate((x_dev_metafeatures[:,:,:-5], x_dev_metafeatures[:,:,-1:]), axis=2)
    #x_dev_metafeatures = np.concatenate((x_dev_stance, x_dev_metafeatures), axis=2)
    
    # Loading the test features
    x_test_embeddings =  np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\sacred-twitter-data-w-test\\test\\embeddings_array.npy")
    x_test_metafeatures = np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\sacred-twitter-data-w-test\\test\\feature_array.npy")

    # Loading the veracity of the test set
    y_test =  np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\sacred-twitter-data-w-test\\test\\labels.npy")

    # Loading the ids of the test set
    ids_test =  np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\sacred-twitter-data-w-test\\test\\tweet_ids.npy", allow_pickle=True)
metafeatures = metafeatures[:,:,-5:-1]
x_train_metafeatures = np.concatenate((x_train_metafeatures[:,:,:-5], x_train_metafeatures[:,:,-1:]), axis=2)
x_train_metafeatures = np.concatenate((x_train_stance, x_train_metafeatures), axis=2)
"""