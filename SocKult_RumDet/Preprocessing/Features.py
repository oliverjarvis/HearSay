import pandas as pd
from SBERT_WK_Embedding import SBERT_WK_Embedding
source_tweets = "data/twitter-english-source-clean-final.csv"
reply_tweets = "data/twitter-english-source-replies-clean-final.csv"
source_df = pd.read_csv(source_tweets, dtype={"id_str":object})
reply_df = pd.read_csv(reply_tweets, dtype={"id_str":object})
datasets = source_df.append(reply_df)

def FeatureFetch(func):
    def get_data_for_tweet_id(id):
        row = datasets.loc[datasets['id_str'] == id]
        return row
    def wrapper(*args):
        #loops through the tweet_ids and fetches the data for each
        branch = args[0]
        ids = []
        for id in branach:
            ids.append(get_data_for_tweet_id(id))
        return func(ids)
    return wrapper

# All features using the feature fetch decorator should now be able to have
# input a variable amount of tweet ids, and have returned an arbitrary feature
# vector performed on.

# Template:
# @FeatureFetch
# def feature_you_want(tweet_id_1, [tweet_id_2, ...]):
#   return feature


# Dependencies
Embedder = SBERT_WK_Embedding()

@FeatureFetch
def embeddings(branch):
    embeddings = []
    for tweet in branch:
        embedding = SBERT_WK_Embedding().get_embeddings(source_tweet)
        embeddings.append(np.asarray(embedding))
    return np.asarray(embeddings)

@FeatureFetch
def cosine_similarity(branch):
    #for parent comment
    cosine_similarities = [0.0]
    for tweet_id in range(1, len(branch)):
        cosine = SBERT_WK_Embedding().get_cosine(branch[0], branch[tweet_id])
        cosine_similarity.append(np.asarray([cosine]))
    return return np.asarray(cosine_similarity)

@FeatureFetch
def column_features(branch):
    features = []
    for tweet in branch:
        features = tweet.to_dict()
        features.append(np.asarray(preprocess(features)))
    return np.asarray(features)

    def scale(value, col):
        return (value - datasets.describe()[col]['mean']) / datasets_decription[col]['std']

    def preprocess(row):
        fd = []
        #function that normalizes the features
        for k, v in row.items():
            if k == "id_str":
                continue
            if k == "user.verified":
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


    



