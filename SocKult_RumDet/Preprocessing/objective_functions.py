"""
This code contains several versions of objective functions to be used together
with parameter search functions
"""
from hyperopt import STATUS_OK
from model import LSTM_model_veracity
from sklearn.metrics import f1_score
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from branch2treelabels import branch2treelabels
from keras.preprocessing.sequence import pad_sequences

#%%
def objective_function_veracity_branchLSTM(params):
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
    mactest_F = f1_score(tree_label, tree_prediction, average='macro')
    output = {'loss': 1-mactest_F,
              'Params': params,
              'status': STATUS_OK,
              'attachments': {'ID':trees,'Predictions':tree_prediction, 'Labels':tree_label}}
    return output