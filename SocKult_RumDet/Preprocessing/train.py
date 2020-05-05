import numpy as np 

with open("saved_dataRumEval2019/dev/tweet_ids.npy") as f:
  foo = f.readlines()
x = np.load(foo)

feature_array = np.load("saved_dataSocKult_RumDet/dev/feature_array.npy", allow_pickle=True)
labels = np.load("saved_dataSocKult_RumDet/dev/labels.npy")
embeddings_array = np.load("saved_dataSocKult_RumDet/dev/labels.npy")
saved_dataSocKult_RumDet/dev/ids.npy
saved_dataSocKult_RumDet/dev/train_array.npy
saved_dataSocKult_RumDet/dev/tweet_ids.npy
x.shape