import os
import argparse
import numpy as np
from pathlib import Path

from Preprocessor.Features import FeatureFetch
from Preprocessor.preprocessing_tweets import load_dataset
from Preprocessor.transform_feature_dict import transform_feature_dict

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
        if type(f_vec) == bool:
            return False
        feature_vector = np.concatenate((feature_vector, f_vec), axis=1)
    return np.asarray(feature_vector)

def prep_pipeline(data_path, dataset, output_path, metacontext_path, featureParser, feature_set):
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
        #if fold=="dev" or fold=="train":
        #    continue
        for conversation in folds[fold]:
            branches = conversation['branches']

            for branch in branches:
                temp_ids = []
                branch_features = feature_vectors_for_branch(branch, feature_set) #should output (branches, branch_length, feature_size)
                if type(branch_features) == bool:
                    continue
                feature_fold.append(branch_features)
                for tweet in branch:
                    temp_ids.append(tweet['id_str'])
                tweet_ids.append(temp_ids)
                labels.append(convert_label(conversation['veracity']))
        
        #convert the whole thing to a numpy array
        feature_fold = np.asarray(feature_fold)

        if feature_fold.size > 0:

            feature_fold = pad_sequences(feature_fold, maxlen=25,
                                         dtype='float32',
                                         padding='post',
                                         truncating='post', value=0.)#feature_fold.shape = [rootnode, branches,]
            tweet_ids = np.asarray(tweet_ids)
            labels = np.asarray(labels)
            path_fold = os.path.join(path, fold)
            if not os.path.exists(path_fold):
                os.makedirs(path_fold)

            embeddings_array = feature_fold[:,:,:EMBEDDING_DIM]
            feature_array = feature_fold[:,:,EMBEDDING_DIM:]

            np.save(os.path.join(path_fold, 'embeddings_array'), embeddings_array)
            np.save(os.path.join(path_fold, 'feature_array'), feature_array)
            np.save(os.path.join(path_fold, 'labels'), labels)
            np.save(os.path.join(path_fold, 'ids'), ids) 
            np.save(os.path.join(path_fold, 'tweet_ids'), tweet_ids)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('HearSay Preprocessing')
    parser.add_argument('--input', '-i', help='folder for training data', default="Data")
    parser.add_argument('--dataset', '-d', help='name of dataset', default="")
    parser.add_argument('--metacontext', '-m', help="hydrated twitter data folder")
    parser.add_argument('--output', '-o', help='destination for files', default="")
    parser = parser.parse_args()
    featurefetch = FeatureFetch(parser.metacontext)
    features = [
        featurefetch.embeddings,
        featurefetch.column_features,
        featurefetch.one_hot_stance
    ]

    prep_pipeline( 
        data_path=parser.input.strip(), 
        dataset = parser.dataset.strip(), 
        output_path=parser.output.strip(),
        metacontext_path=parser.metacontext.strip(),
        featureParser = featurefetch,
        feature_set=features,
    )
    