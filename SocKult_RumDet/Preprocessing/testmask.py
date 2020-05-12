import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import TimeDistributed, Masking, LSTM
emb = np.array([[[83, 91, 1, 645, 1253, 927], [73, 0, 0, 0, 0], [0, 0, 71], [0, 0], []],
  [
  [83, 91, 1, 645, 1253, 927],
  [73, 0, 0, 0, 0],
  [0, 0, 71],
  [0, 0],
  []
  ]
])
meta = np.array([
  [
  [83, 91, 1, 645, 1253, 927],
  [73, 0, 0, 0, 0],
  [0, 0, 71],
  [0, 0],
  []
  ],
  [
    [83, 91, 1, 645, 1253, 927],
  [73, 0, 0, 0, 0],
  [0, 0, 71],
  [0, 0],
  []
  ]
])

# Loading the dev features (even though we still call it test, might change)
emb =  np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\sacred-twitter-data-w-test\\dev\\embeddings_array.npy")
meta = np.load("C:\\Users\\sysadmin\\Downloads\\HearSay\\SocKult_RumDet\\Preprocessing\\sacred-twitter-data-w-test\\dev\\feature_array.npy")



print(meta.shape)

                



#emb_input = Input(shape = emb_shape, name = 'Embeddings')
#metafeatures_input = Input(shape = metafeatures_shape, name = 'Metafeatures')

# Adding masks to account for zero paddings
emb_mask = Masking(mask_value=0, input_shape = (None, emb.shape))(emb)
metafeatures_mask = Masking(mask_value=0, input_shape = (None, meta.shape))(meta)

emb_LSTM_query = LSTM(100, dropout=0.5, recurrent_dropout=0.5,
                        return_sequences=True)(emb_mask)

metafeatures_LSTM_query = LSTM(100, dropout=0.5, recurrent_dropout=0.5,
                            return_sequences=True)(metafeatures_mask)

print(emb_LSTM_query._keras_mask[0])
print(metafeatures_LSTM_query._keras_mask)
