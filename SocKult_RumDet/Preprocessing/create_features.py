from SBERT_WK_Embedding import SBERT_WK_Embedding
import pandas as pd
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import copy
#from tree2branches import tree2branches

# Loading the data
source_tweets = "SocKult_RumDet/Preprocessing/data/twitter-english-source-clean-final.csv"
reply_tweets = "SocKult_RumDet/Preprocessing/data/twitter-english-source-replies-clean-final.csv"

source_df = pd.read_csv(source_tweets)
reply_df = pd.read_csv(reply_tweets)
datasets_decription = source_df.append(reply_df).describe()

Embedder = SBERT_WK_Embedding()

def correlation_heatmap(train):
    correlations = train.corr()

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show();

def EDA(source_df, reply_df):
    """function that does exploratory data analysis on the data"""

    # Creating profile report
    source_report = ProfileReport(source_df, title='Profile Report', html={'style':{'full_width':True}})
    source_report.to_notebook_iframe()
    source_report.to_file(output_file="EDA_source_report.html")

    reply_report = ProfileReport(reply_df, title='Profile Report', html={'style':{'full_width':True}})
    reply_report.to_notebook_iframe()
    reply_report.to_file(output_file="EDA_reply_report.html")

    correlation_heatmap(source_df)
    correlation_heatmap(reply_df)

    import pdfkit 
    pdfkit.from_file('EDA_source_report.html', 'EDA_source_report.pdf') 
    pdfkit.from_file('EDA_reply_report.html', 'EDA_reply_report.pdf') 

def scale(value, col):
    return (value - datasets_decription[col]['mean']) / datasets_decription[col]['std']

def preprocess(row, feature_dict):
    fd = copy.deepcopy(feature_dict)
    #function that normalizes the features
    for col in row:
        if col == "user.verified":
            if row[col] == "True":
                fd[col] = 1
            else:
                fd[col] == 0
        else:
            fd[col] == scale(row[col], col)
    return fd

def metadata_for_tweet_id(tweet_id, tweet_type):
    source_file = ""
    if tweet_type == "source":
        source_file = pd.read_csv(source_tweets, header=0)
    elif tweet_type == "reply":
        source_file = pd.read_csv(reply_tweets, header=0)

    try:
       row = source_file.loc[source_file['id_str'] == tweet_id]
    except:
        print("Row for str_id: [{}] missing from data".format(tweet_id))
        print("Unsure how to proceed.")    

    features = row.to_dict()  
    feature_dict = {}
    # features should really only output one row, but we'll extract them as a precaution
    for k, v in features.items():
        feature_dict[k] = v.values()[0]
    
    #preprocessing for meta data features
    feature_dict = preprocess(row, feature_dict)

    return feature_dict

def get_feature_vector(conversation):
    # get the embeddings from
    source = conversation['source']
    source_tweet = source['text']
    source_id = source['id_str']
    source_features = {}
    source_features['SBERT-WK'] = SBERT_WK_Embedding().get_embeddings(source_tweet)
    source_features['metadata'] = metadata_for_tweet_id(tweet_id=source_id, tweet_type="source")
    source_features['issource'] = 1

    fullthread_featdict = {}
    fullthread_featdict[source['id_str']] = source_features
    for tw in conversation['replies']:
        feature_dict = {}
        feature_dict['issource'] = 0
        #This is where the embeddings are added 
        tweet_id = tw['id_str']
        feature_dict['SBERT-WK'] = Embedder.get_embeddings(tw['text'])
        feature_dict['metadata'] = metadata_for_tweet_id(tweet_id=tweet_id, tweet_type="reply")
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