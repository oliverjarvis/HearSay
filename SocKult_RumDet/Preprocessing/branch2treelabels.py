"""
This is a postprocessing function that takes per-branch predicitons and takes 
majority vote to generate per-tree prediction
"""
import numpy as np
from keras.preprocessing.sequence import pad_sequences

def branch2treelabels(ids_test, y_test, y_pred, confidence):
    trees = np.unique(ids_test)
    trees = pad_sequences(trees, maxlen=None,
                                         dtype='float32',
                                         padding='post',
                                         truncating='post', value=0)
    ids_test = pad_sequences(ids_test, maxlen=None,
                                         dtype='float32',
                                         padding='post',
                                         truncating='post', value=0)
    #ids_test = ids_test.tolist()
    #trees = trees.tolist()
    tree_prediction = []
    tree_label = []
    tree_confidence = []
    for tree in trees:
        tree_idx = np.where(ids_test == tree)[0]
        tree_label.append(y_test[tree_idx])
        tree_confidence.append(confidence[tree_idx])
        temp_prediction = [y_pred[i] for i in tree_idx]
        # all different predictions from branches from one tree
        unique, counts = np.unique(temp_prediction, return_counts=True)
        tree_prediction.append(unique[np.argmax(counts)])
    return trees, tree_prediction, tree_label, tree_confidence

def branch2treelabels_test(ids_test, y_pred, confidence):
    trees = np.unique(ids_test)
    tree_prediction = []
    tree_confidence = []
    for tree_idx in range(len(trees)):
        #treeindx = np.where(ids_test == tree)[0]
        tree_confidence.append(np.float(confidence[tree_idx]))
        temp_prediction = [y_pred[i] for i in tree_idx]
        # all different predictions from branches from one tree
        unique, counts = np.unique(temp_prediction, return_counts=True)
        tree_prediction.append(unique[np.argmax(counts)])
    return trees, tree_prediction, tree_confidence