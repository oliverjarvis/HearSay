"""
This is a postprocessing function that takes per-branch predicitons and takes 
majority vote to generate per-tree prediction
"""
import numpy as np

from keras.preprocessing.sequence import pad_sequences

def branch2treelabels(ids_test, y_test, y_pred, confidence):
    ops = np.array([list[0] for list in ids_test])
    trees = np.unique(ops)
    tree_prediction = []
    tree_label = []
    tree_confidence = []

    for tree in trees:
        treeindx = np.where(ops == tree)[0]
        tree_label.append(y_test[treeindx[0]])
        tree_confidence.append(confidence[treeindx[0]])
        temp_prediction = [y_pred[i] for i in treeindx]
        # all different predictions from branches from one tree
        unique, counts = np.unique(temp_prediction, return_counts=True)
        tree_prediction.append(unique[np.argmax(counts)])
    return trees, tree_prediction, tree_label, tree_confidence



def branch2treelabels_test(ids_test, y_pred, confidence):
    trees = np.unique(ids_test)
    tree_prediction = []
    tree_confidence = []
    for tree_idx in range(len(trees)):
        tree_confidence.append(np.float(confidence[tree_idx]))
        temp_prediction = [y_pred[i] for i in tree_idx]
        # all different predictions from branches from one tree
        unique, counts = np.unique(temp_prediction, return_counts=True)
        tree_prediction.append(unique[np.argmax(counts)])
    return trees, tree_prediction, tree_confidence
