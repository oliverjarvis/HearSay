import pandas as pd

df = pd.read_csv("data/twitter-english-source-clean.csv")
df2 = pd.read_csv("data/twitter-english-source-replies-clean.csv")

df = df.drop(["retweeted", "favorited"], axis = 1)
df2 = df2.drop(["retweeted", "favorited"], axis = 1)

df.to_csv("data/twitter-english-source-clean-final.csv")
df2.to_csv("data/twitter-english-source-replies-clean-final.csv")