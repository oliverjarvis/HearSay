import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed, Masking
emb = [
  [83, 91, 1, 645, 1253, 927],
  [73, 8, 0, 55, 927],
  [711, 632, 71],
  [],
  []
]

meta = [
  [83, 91, 1, 645, 1253, 927],
  [73, 0, 0, 0, 0],
  [0, 0, 71],
  [0, 0],
  []
]


padded_emb = tf.keras.preprocessing.sequence.pad_sequences(emb,
                                                              padding='post')


padded_meta = tf.keras.preprocessing.sequence.pad_sequences(meta,
                                                              padding='post')
                



#emb_input = Input(shape = emb_shape, name = 'Embeddings')
#metafeatures_input = Input(shape = metafeatures_shape, name = 'Metafeatures')

# Adding masks to account for zero paddings
emb_mask = Masking(mask_value=0, input_shape = (None, padded_emb.shape))(padded_emb)
metafeatures_mask = Masking(mask_value=0, input_shape = (None, padded_meta.shape))(padded_meta)
print(emb_mask._keras_mask)
print(metafeatures_mask._keras_mask)