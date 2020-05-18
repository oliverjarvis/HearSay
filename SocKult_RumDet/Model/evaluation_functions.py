#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code contains several versions of evaluation functions
"""
import numpy as np
import os

from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from keras.utils.np_utils import to_categorical

import pickle
from copy import deepcopy
from pathlib import Path

from Model.model import LSTM_model_veracity

from Utils.branch2treelabels import branch2treelabels


embeddings_name = "embeddings_array.npy"
metafeatures_name = "feature_array.npy"
labels_name = "labels.npy"
ids_name = "tweet_ids.npy"

#event_splits = pickle.load(open("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\sacred-twitter-data-w-test\\splits.pickle", "rb"))


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
# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')

  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()

def load_data_from_dir(dir, num_classes=3, with_ids=False, data_dir = None):
    embeddings = np.load( data_dir / dir / embeddings_name )
    metafeatures = np.load( data_dir / dir / metafeatures_name )
    labels = np.load( data_dir / dir / labels_name)
    if num_classes > 0:
        labels = to_categorical(labels, num_classes=num_classes)
    if with_ids:
        ids = np.load( data_dir / dir / ids_name, allow_pickle=True)
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

def examine_event(data, tweet_ids, event, etype):
    event_ids = []
    for k, v in event_splits.items():
        if k == event:
            event_ids.extend(v)
    valid_rows = []
    row_idx = 0
    for row in tweet_ids:
        if row[0] in event_ids:
            valid_rows.append(data[row_idx])
        row_idx += 1
    valid_rows = np.array(valid_rows, dtype="object")
    np.save("{}_{}.np".format(event, etype), valid_rows)


def split_event_loo(data, tweet_ids, event):
    event_ids = []
    for k, v in event_splits.items():
        if k != event:
            event_ids.extend(v)
    event_ids = np.array(event_ids, dtype="object")
    train_tweet_ids = []
    dev_tweet_ids = []
    tweet_id_idx = 0
    for row in tweet_ids:
        if row[0] in event_ids:
            train_tweet_ids.append(tweet_id_idx)
        else:
            dev_tweet_ids.append(tweet_id_idx)
        tweet_id_idx += 1
    data_train = np.delete(data, dev_tweet_ids, axis=0)
    data_dev = np.delete(data, train_tweet_ids, axis=0)
    return data_train, data_dev

#%%
def evaluation_function_veracity_branchLSTM(data_dir, params, metac, use_embeddings=True, event="", Early_Stopping = True):

    x_train_embeddings, x_train_metafeatures, y_train, ids_train = load_data_from_dir("train", num_classes=3, with_ids=True, data_dir = data_dir)
    x_dev_embeddings, x_dev_metafeatures, y_dev, ids_dev = load_data_from_dir("dev", num_classes=3, with_ids=True, data_dir = data_dir)
    x_test_embeddings, x_test_metafeatures, y_test, ids_test = load_data_from_dir("test", num_classes=0, with_ids=True, data_dir = data_dir)

    x_train_metafeatures = split_features(x_train_metafeatures, metac)
    x_dev_metafeatures = split_features(x_dev_metafeatures, metac)
    x_test_metafeatures = split_features(x_test_metafeatures, metac)

    x_train_embeddings = np.concatenate((x_train_embeddings, x_dev_embeddings), axis = 0)
    x_train_metafeatures = np.concatenate((x_train_metafeatures, x_dev_metafeatures), axis = 0)
    y_train = np.concatenate((y_train, y_dev), axis=0)
    ids_train = np.concatenate((ids_train, ids_dev), axis=0)

    #x_train_embeddings, x_dev_embeddings = split_event_loo(x_train_embeddings, ids_train, event)
    #x_train_metafeatures, x_dev_metafeatures = split_event_loo(x_train_metafeatures, ids_train, event)
    #y_train, y_dev = split_event_loo(y_train, ids_train, event)

    # Getting predictions and confidence of the model
    y_pred, confidence = LSTM_model_veracity(x_train_embeddings, x_train_metafeatures, y_train,
                                        x_test_embeddings, x_test_metafeatures, params, use_embeddings=use_embeddings, Early_Stopping=Early_Stopping)
    
    #Getting the predictions of trees and the branches
    trees, tree_prediction, tree_label, tree_confidence = branch2treelabels(ids_test, 
                                                            y_test,
                                                            y_pred,
                                                            confidence)
    
    mactest_F = f1_score(tree_label, tree_prediction, average='macro')
            
    return trees, tree_prediction, tree_label, tree_confidence, mactest_F