import os
import pandas as pd 
from pathlib import path

csv_dir = 

csv_dir = Path(csv_dir)

train_df <- pd.read_csv(csv_dir/'train-key.csv')


train_df = train_df[['id', 
'favorite_count', 
'retweet_count', 
'user_verified',
'user_followers_count',
'user_listed_count',
'user_friends_count',
'user_favourites_count']]
