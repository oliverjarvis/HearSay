"""
This code helps convert dictionaries of features from conversation into
arrays of branches of conversation
"""

import numpy as np
from tree2branches import tree2branches
import tensorflow as tf

#%%
def convert_label(label):
    # One-hot encoding the stance labels
    labels = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
    if label == "support":
        return(labels[0])
    elif label == "comment":
        return(labels[1])
    elif label == "deny":
        return(labels[2])
    elif label == "query":
        return(labels[3])
    else:
        print(label)
#%%
def transform_feature_dict(thread_feature_dict, conversation, feature_set):
    thread_features_array = []
    thread_stance_labels = []
    clean_branches = []

    branches = conversation['branches']

    for branch in branches:
        branch_rep = []
        clb = []
        branch_stance_lab = []
        for twid in branch:
            if twid in thread_feature_dict.keys():
                tweet_rep = dict_to_array(thread_feature_dict[twid],
                                          feature_set)

                if twid == branch[0]:
                    if 'label' in list(conversation['source'].keys()):
                        branch_stance_lab = convert_label(
                            conversation['source']['label'])
                        tweet_rep.extend(branch_stance_lab)
                        tweet_rep = np.asarray(tweet_rep)
                    clb.append(twid)
                else:
                    for r in conversation['replies']:
                        if r['id_str'] == twid:
                            if 'label' in list(r.keys()):
                                branch_stance_lab = convert_label(
                                    r['label'])
                                tweet_rep.extend(branch_stance_lab)
                                tweet_rep = np.asarray(tweet_rep)
                            clb.append(twid)
            branch_rep.append(tweet_rep)

        if branch_rep != []:
            branch_rep = np.asarray(branch_rep)
            #branch_stance_lab = np.asarray(branch_stance_lab)
            thread_features_array.append(branch_rep)
            #thread_features_array.extend(branch_stance_lab)
            #thread_stance_labels.append(branch_stance_lab)
            clean_branches.append(clb)
     
    return thread_features_array, clean_branches #thread_stance_labels, 

feature_keys = []

#%%
def dict_to_array(feature_dict, feature_set):
    global feature_keys
    tweet_rep = []
    for feature_name in feature_set:
        if type(feature_dict[feature_name]) == dict:
            for k in feature_dict[feature_name].keys():
                if feature_keys == []:
                    feature_keys = feature_dict[feature_name].keys()
                else:
                    if feature_keys != feature_dict[feature_name].keys():
                        print("big doodoo")
                if np.isscalar(feature_dict[feature_name][k]):
                    tweet_rep.append(feature_dict[feature_name][k])
                else:
                    tweet_rep.extend(feature_dict[feature_name][k])
        else:
            if np.isscalar(feature_dict[feature_name]):
                tweet_rep.append(feature_dict[feature_name])
            else:
                tweet_rep.extend(feature_dict[feature_name])
    return tweet_rep

"""feature _keys = =  []""issource, ""favorite_cout
                    nnt, ""retweet_cou
                    nt, ""user.verifi
                    ed, 
                    "user.followers_count",
                    "user.listed_count",
                    "user.friends_count",
                    "user.favourites_count"""