from preprocessing_tweets import load_dataset
from transform_feature_dict import transform_feature_dict
import help_prep_functions
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
#from create_features import get_feature_vector
import Features

embedding_dimension = 768

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

def prep_pipeline(dataset, feature_set):
    #regular old start
    path = 'saved_data'+dataset
    folds = {}
    folds = load_dataset()
    
    #Olivercentric
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
            path_fold = os.path.join(path, fold)
            if not os.path.exists(path_fold):
                os.makedirs(path_fold)

            embeddings_array = feature_fold[:,:,:embedding_dimension]
            metafeatures_array = feature_fold[:,:,embedding_dimension:]

            np.save(os.path.join(path_fold, 'embeddings_array'), embeddings_array)
            np.save(os.path.join(path_fold, 'metafeatures_array'), metafeatures_array)
            np.save(os.path.join(path_fold, 'labels'), labels)
            #np.save(os.path.join(path_fold, 'ids'), ids) 
            np.save(os.path.join(path_fold, 'tweet_ids'), tweet_ids)
      
def main(data ='RumEval2019'):
    features = [
        Features.embeddings,
        Features.column_features,
        Features.one_hot_stance
    ]
    prep_pipeline(dataset='newdata', feature_set=features)

if __name__ == '__main__':
    main()