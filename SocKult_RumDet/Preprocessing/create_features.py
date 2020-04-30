from SBERT_WK_Embedding import SBERT_WK_Embedding
import pandas as pd
#from tree2branches import tree2branches


source_tweets = "SocKult_RumDet/Preprocessing/data/twitter-english-source-clean.csv"
reply_tweets = "SocKult_RumDet/Preprocessing/data/twitter-english-replies-clean.csv"

Embedder = SBERT_WK_Embedding()

def metadata_for_tweet_id(tweet_id, tweet_type):
    source_file = ""
    if tweet_type == "source":
        source_file = pd.read_csv(source_tweets, header=0)
    elif tweet_type == "reply":
        source_file = pd.read_csv(reply_tweets, header=0)

    try:
       row = source_file.loc[source_file['str_id'] == tweet_id]
    except:
        print("Row for str_id: [{}] missing from data".format(tweet_id))
        print("Unsure how to proceed.")    

    features = row.to_dict()  
    feature_dict = {}
    # features should really only output one row, but we'll extract them as a precaution
    for k, v in features.items():
        feature_dict[k] = v.values()[0]
    
    #preprocessing for meta data features
    # 😃

    return feature_dict

def get_feature_vector(conversation):
    # get the embeddings from
    source = conversation['source']
    source_tweet = source['text']
    source_id = source['id_str']
    source_features = {}
    source_features['SBERT-WK'] = SBERT_WK_Embedding().get_embeddings(source_tweet)
    #source_features['metadata'] = metadata_for_tweet_id(tweet_id=source_id, tweet_type=TweetTypes.source)
    source_features['issource'] = 1

    fullthread_featdict = {}
    fullthread_featdict[source['id_str']] = source_features
    for tw in conversation['replies']:
        feature_dict = {}
        feature_dict['issource'] = 0
        #This is where the embeddings are added 
        feature_dict['SBERT-WK'] = Embedder.get_embeddings(tw['text'])
        fullthread_featdict[tw['id_str']] = feature_dict

    return fullthread_featdict


# similarity scores
# if we want to calculate these we will have to go through
# the same looping procedure as originally implemented
"""
feature_dict['src_unconfirmed'] = 0
feature_dict['src_rumour'] = 0
feature_dict['thread_unconfirmed'] = 0
feature_dict['thread_rumour'] = 0

if 'unconfirmed' in tokens:
    feature_dict['src_unconfirmed'] = 1
if 'unconfirmed' in otherthreadtokens:
    feature_dict['thread_unconfirmed'] = 1
if 'rumour' in tokens or 'gossip' in tokens or 'hoax' in tokens:
    feature_dict['src_rumour'] = 1
if ('rumour' in otherthreadtokens) or ('gossip' in otherthreadtokens) or ('hoax' in otherthreadtokens):
    feature_dict['thread_rumour'] = 1
"""   