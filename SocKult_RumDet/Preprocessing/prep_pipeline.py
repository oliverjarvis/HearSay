"""
This is outer preprocessing file

To run:
    
python prep_pipeline.py

Main function has parameter that can be changed:

feats ('text' or 'SemEval')

"""
from preprocessing_tweets import load_dataset
from transform_feature_dict import transform_feature_dict
#from extract_thread_features import extract_thread_features_incl_response
import help_prep_functions
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
from create_features import get_feature_vector
#%%

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

def prep_pipeline(dataset='RumEval2019', feature_set=['avgw2v']):
    
    path = 'saved_data'+dataset
    folds = {}
    folds = load_dataset()
    if 'avgw2v' in feature_set:
        help_prep_functions.loadW2vModel()
    
    for fold in folds.keys():
        print(fold)
        feature_fold = []
        tweet_ids = []
        fold_stance_labels = []
        labels = []
        ids = []
        for conversation in folds[fold]:

            thread_feature_dict = get_feature_vector(conversation)

            thread_features_array, branches = transform_feature_dict(
                                   thread_feature_dict, conversation,
                                   feature_set=feature_set)
            #thread_stance_labels, 
            #fold_stance_labels.extend(thread_stance_labels)
            tweet_ids.extend(branches)
            feature_fold.extend(thread_features_array)
            for i in range(len(thread_features_array)):
                labels.append(convert_label(conversation['veracity']))
                ids.append(conversation['id'])
            
        if feature_fold!=[]:

            feature_fold = pad_sequences(feature_fold, maxlen=None,
                                         dtype='float32',
                                         padding='post',
                                         truncating='post', value=0.)#feature_fold.shape = [rootnode, branches,]
    
            #fold_stance_labels = pad_sequences(fold_stance_labels, maxlen=None,
            #                                   dtype='float32',
            #                                   padding='post', truncating='post',
            #                                   value=0.)
            #one hot
            labels = np.asarray(labels)
            path_fold = os.path.join(path, fold)
            if not os.path.exists(path_fold):
                os.makedirs(path_fold)

            embeddings_array = feature_fold[:,:,:embedding_dimension]
            feature_array = feature_fold[:,:,embedding_dimension:]

            np.save(os.path.join(path_fold, 'embeddings_array'), embeddings_array)
            np.save(os.path.join(path_fold, 'feature_array'), feature_array)
            np.save(os.path.join(path_fold, 'labels'), labels)
            #np.save(os.path.join(path_fold, 'fold_stance_labels'),
            #        fold_stance_labels)
            np.save(os.path.join(path_fold, 'ids'), ids) 
            np.save(os.path.join(path_fold, 'tweet_ids'), tweet_ids)
      
def main(data ='RumEval2019', feats = 'SocKultfeatures'):

    if feats == 'text':
        features = ['avgw2v']
    elif feats == 'SemEvalfeatures':
        features = ['avgw2v', 'hasnegation', 'hasswearwords',
                           'capitalratio', 'hasperiod', 'hasqmark',
                           'hasemark', 'hasurl', 'haspic',
                           'charcount', 'wordcount', 'issource',
                           'Word2VecSimilarityWrtOther',
                           'Word2VecSimilarityWrtSource',
                           'Word2VecSimilarityWrtPrev']
    elif feats == 'SocKultfeatures':
        features = ['SBERT-WK', 'issource', 'metadata']
    else:
        print("Features not supported")
        return
    
    prep_pipeline(dataset='SocKult_RumDet', feature_set=features)

if __name__ == '__main__':
    main()