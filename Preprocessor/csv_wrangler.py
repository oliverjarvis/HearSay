import os
import pandas as pd 
from pathlib import Path

def column_cleaner(data_dir):
    data_dir = Path(data_dir)

    train_df = pd.read_csv(data_dir/'train-key.csv')
    dev_df = pd.read_csv(data_dir/'dev-key.csv')
    test_df = pd.read_csv(data_dir/'final-eval-key.csv')

    col_names = ['id', 
    'favorite_count', 
    'retweet_count', 
    'user_verified',
    'user_followers_count',
    'user_listed_count',
    'user_friends_count',
    'user_favourites_count']

    train_df = train_df[col_names]
    dev_df = dev_df[col_names]
    test_df = test_df[col_names]

    train_df.rename(columns = {'id': 'id_str'}, inplace = True)
    dev_df.rename(columns = {'id': 'id_str'}, inplace = True)
    test_df.rename(columns = {'id': 'id_str'}, inplace = True) 

    train_df.to_csv(data_dir/'final-train-key-clean.csv', index = False)
    dev_df.to_csv(data_dir/'final-dev-key-clean.csv', index = False)
    test_df.to_csv(data_dir/'final-test-key-clean.csv', index = False)

column_cleaner("/Users/au578822/Desktop/Oliver/Random/HearSay/Data")