import numpy as np 

# %%
feature_array = np.load("saved_dataSocKult_RumDet/dev/feature_array.npy", allow_pickle=True)
labels = np.load("saved_dataSocKult_RumDet/dev/labels.npy", allow_pickle=True)
embeddings_array = np.load("saved_dataSocKult_RumDet/dev/labels.npy", allow_pickle=True)
ids = np.load("saved_dataSocKult_RumDet/dev/ids.npy", allow_pickle=True)
train_array = np.load("saved_dataSocKult_RumDet/dev/train_array.npy", allow_pickle=True)
tweet_ids = np.load("saved_dataSocKult_RumDet/dev/tweet_ids.npy", allow_pickle=True)

# %%
print(feature_array.shape)
print(labels.shape)
print(embeddings_array.shape)
print(ids.shape)
print(train_array.shape)
print(tweet_ids.shape)

# %%
feature_array[1][0]

# %%
