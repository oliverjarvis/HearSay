import os
import argparse
import numpy as np
from pathlib import Path

from Preprocess.Features import Features
from .Preprocess.preprocessing_tweets import load_dataset
from .Preprocess.transform_feature_dict import transform_feature_dict

from keras.preprocessing.sequence import pad_sequences

EMBEDDING_DIM = 768

def convert_label(label):
    if label == "true":
        return(0)
    elif label == "false":
        return(1)
    elif label == "unverified":
        return(2)
    else:
        print(label)

def feature_vectors_for_branch(branch, features):
    feature_vector = np.empty([len(branch), 0])
    for feature in features:
        f_vec = feature(branch) #(branche_length, feature_size)
        feature_vector = np.concatenate((feature_vector, f_vec), axis=1)
    return np.asarray(feature_vector)

def prep_pipeline(data_path, dataset, output_path, metacontext_path, feature_set):
    #regular old start
    path = Path(output_path)
    folds = {}
    folds = load_dataset(data_path, dataset)

    for fold in folds.keys():
        print(fold)
        feature_fold = []
        tweet_ids = []
        fold_stance_labels = []
        labels = []
        ids = []
        
        for conversation in folds[fold]:
            branches = conversation['branches']
            for branch in branches:
                temp_ids = []
                branch_features = feature_vectors_for_branch(branch, feature_set) #should output (branches, branch_length, feature_size)
                feature_fold.append(branch_features)
                for tweet in branch:
                    temp_ids.append(tweet['id_str'])
                tweet_ids.append(temp_ids)
                labels.append(convert_label(conversation['veracity']))
        
        #convert the whole thing to a numpy array
        feature_fold = np.asarray(feature_fold)

        if feature_fold!=[]:

            feature_fold = pad_sequences(feature_fold, maxlen=25,
                                         dtype='float32',
                                         padding='post',
                                         truncating='post', value=0.)#feature_fold.shape = [rootnode, branches,]
            tweet_ids = np.asarray(tweet_ids)
            labels = np.asarray(labels)
            path_fold = path / fold
            if not path_fold.is_dir():
                path_fold.mkdir()

            embeddings_array = feature_fold[:,:,:EMBEDDING_DIM]
            metafeatures_array = feature_fold[:,:,EMBEDDING_DIM:]

            np.save(path_fold / 'embeddings_array', embeddings_array)
            np.save(path_fold / 'metafeatures_array', metafeatures_array)
            np.save(path_fold / 'labels', labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('HearSay Preprocessing')
    parser.add_argument('--input', '-i', help='folder for training data', default="Data")
    parser.add_argument('--dataset', '-d', help='name of dataset', default="")
    parser.add_argument('--metacontext', '-m', help="hydrated twitter data folder")
    parser.add_argument('--output', '-o', help='destination for files', default="")

    features = [
        Features.embeddings,
        Features.column_features,
        Features.one_hot_stance
    ]

    prep_pipeline( 
        data_path=parser.input, 
        dataset = parser.dataset, 
        output_path=parser.output,
        metacontext_path=parser.metacontext,
        feature_set=features
    )
    