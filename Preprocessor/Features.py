import os
import numpy as np
import pandas as pd
from glob import glob
import tensorflow as tf
from pathlib import Path
from Utils.SBERT_WK_Embedding import SBERT_WK_Embedding

'''
def FeatureFetch(func):
    def wrapper(*args):
        branch = args[0]
        return func(branch)
    return wrapper

class Featurizer:

        self.Embedder = SBERT_WK_Embedding()
        self.Embedding_cache = {}
'''
Embedding_cache = {}
Embedder = SBERT_WK_Embedding()

class FeatureFetch:
    def __init__(self, metacontext_path):
        metacontext_path = metacontext_path.strip()
        files = glob(metacontext_path.strip()+"*.csv")
        for f in files:
            print(f)
        datasets = pd.DataFrame()
        for f in files:
            temp_df =  pd.read_csv(f, dtype={"id_str":object})
            datasets = datasets.append(temp_df)
        self.datasets = datasets

    def embeddings(self, branch):
        embeddings = []
        for tweet in branch:
            if tweet['id_str'] in Embedding_cache:
                embedding = Embedding_cache[tweet['id_str']]
            else:
                embedding = Embedder.get_embeddings(tweet['text'])
                Embedding_cache[tweet['id_str']] = embedding
            embeddings.append(np.asarray(embedding))
        return np.asarray(embeddings)

    def cosine_similarity(self, branch):
        def get_cosine(embedding1, embedding2):
            return embedding1.dot(embedding1) / np.linalg.norm(embedding1) / np.linalg.norm(embedding2)

        #for parent comment
        cosine_similarities = [np.asarray([1.0])]
        if branch[0]['id_str'] in Embedding_cache:
            embedding_op = Embedding_cache[branch[0]['id_str']]
        else:
            embedding_op = Embedder.get_embeddings(branch[0]['text'])
        for tweet_id in range(1, len(branch)):
            #need to implemennt get_cosine
            if branch[tweet_id]['id_str'] in Embedding_cache:
                embedding_tweet = Embedding_cache[branch[tweet_id]['id_str']]
            else:
                embedding_tweet = Embedder.get_embeddings(branch[tweet_id]['text'])

            cosine = get_cosine(embedding_op, embedding_tweet)
            cosine_similarities.append(np.asarray([cosine]))

        return np.asarray(cosine_similarities)

    def one_hot_stance(self, branch):
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
        features = []
        for tweet in branch:
            features.append(convert_label(tweet['label']))
        return np.asarray(features)

    def column_features(self, branch):
        def scale(value, col):
            return (value - self.datasets.describe()[col]['mean']) / self.datasets.describe()[col]['std']

        def preprocess(row):
            fd = []
            #function that normalizes the features
            for k, v in row.items():
                if len(v.values()) < 1:
                    return False
                v = list(v.values())[0]
                if "Unnamed" in k:
                    continue
                if k == "id_str":
                    continue
                if k == "user_verified":
                    if v == True:
                        fd.append(1)
                    else:
                        fd.append(0)
                else:
                    try:
                        fd.append(scale(v, k))
                    except IndexError:
                        print("Oliver er en konge")
                    except:
                        print("We are in some deep shit")
            return np.asarray(fd)
        features = []
        for tweet in branch:
            tweet = self.get_data_for_tweet_id(tweet['id_str'])
            feature_dict = tweet.to_dict()
            a_feature = preprocess(feature_dict)
            if type(a_feature) == bool:
                return False
            else:
                features.append(np.asarray(a_feature))
        return np.asarray(features)

    def get_data_for_tweet_id(self, id):
        row = self.datasets.loc[self.datasets['id_str'] == id]
        return row