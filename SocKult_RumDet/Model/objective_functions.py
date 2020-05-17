"""
This code contains several versions of objective functions to be used together
with parameter search functions
"""
import numpy as np
import os
from pathlib import Path

from hyperopt import STATUS_OK

from sklearn.metrics import f1_score
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

from model import LSTM_model_veracity
from .Utils.branch2treelabels import branch2treelabels


data_dir = Path("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\sacred-twitter-data-w-test\\")
embeddings_name = "embeddings_array.npy"
metafeatures_name = "feature_array.npy"
labels_name = "labels.npy"
ids_name = "tweet_ids.npy"

def load_data_from_dir(dir, num_classes=3, with_ids=False):
    embeddings = np.load( data_dir / dir / embeddings_name )
    metafeatures = np.load( data_dir / dir / metafeatures_name )
    labels = np.load( data_dir / dir / labels_name)
    if num_classes > 0:
        labels = to_categorical(labels, num_classes=num_classes)
    if with_ids:
        ids = np.load( data_dir / dir / ids_name, allow_pickle=True)
        return (embeddings, metafeatures, labels, ids)
    return (embeddings, metafeatures, labels)

#%%
def objective_function_veracity_branchLSTM(params):
    
    x_train_embeddings, x_train_metafeatures, y_train = load_data_from_dir("train", num_classes=3)
    x_test_embeddings, x_test_metafeatures, y_test, ids_test = load_data_from_dir("dev", num_classes=0, with_ids = True)

    # Getting predictions and confidence of the model
    y_pred, confidence = LSTM_model_veracity(x_train_embeddings, x_train_metafeatures, y_train,
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