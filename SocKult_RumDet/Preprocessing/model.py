"""
Contains function that defines the model architecture
"""
import numpy as np
import random
import os
from tensorflow import random as tfr
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, AdditiveAttention, LeakyReLU, LayerNormalization, Bidirectional
from tensorflow.keras.layers import TimeDistributed, Masking
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from tensorflow.keras.utils import plot_model
import json
import datetime

seed_value = 1
os.environ["PYTHONHASHSEED"] = str(0)
random.seed(seed_value)
np.random.seed(seed_value)
tfr.set_seed(seed_value)


log_dir = "SocKult_RumDet\\Preprocessing\\logs\\fit\\" + datetime.datetime.now().strftime("%d%m%Y-%H%M%S")

#%%
def LSTM_model_veracity(x_train_embeddings, x_train_metafeatures, y_train, x_test_embeddings, x_test_metafeatures, params,eval=False, use_embeddings=True, use_metafeatures=True, Early_Stopping=True):
    # Parameter search
    num_lstm_units = int(params['num_lstm_units'])
    num_lstm_layers = int(params['num_lstm_layers'])
    num_dense_layers = int(params['num_dense_layers'])
    num_dense_units = int(params['num_dense_units'])
    num_epochs = params['num_epochs']
    learn_rate = params['learn_rate']
    mb_size = params['mb_size']
    l2reg = params['l2reg']
    dropout = params['dropout']
    attention = params['attention']

    
    # Defining input shapes
    if use_embeddings:
        emb_shape = x_train_embeddings[0].shape

    if use_metafeatures:
        metafeatures_shape = x_train_metafeatures[0].shape

    # Creating the two inputs
    if use_embeddings:
        emb_input = Input(shape = emb_shape, name = 'Embeddings')
    
    if use_metafeatures:
        metafeatures_input = Input(shape = metafeatures_shape, name = 'Metafeatures')

    # Adding masks to account for zero paddings
    if use_embeddings:
        emb_mask = Masking(mask_value=0, input_shape=(None, emb_shape))(emb_input)
    if use_metafeatures:
        metafeatures_mask = (Masking(mask_value=0, input_shape=(None, metafeatures_shape)))(metafeatures_input)

    # Adding attention and LSTM layers with varying layers and units using parameter search            
    if attention == 1:
        for nl in range(num_lstm_layers):
            if use_embeddings:
                emb_LSTM_query = Bidirectional(LSTM(num_lstm_units, dropout=dropout, recurrent_dropout=0.2,
                            return_sequences=True))(emb_mask)

                emb_LSTM_value = Bidirectional(LSTM(num_lstm_units, dropout=dropout, recurrent_dropout=0.2,
                            return_sequences=True))(emb_mask)
            if use_metafeatures:
                metafeatures_LSTM_query = Bidirectional(LSTM(num_lstm_units, dropout=dropout, recurrent_dropout=0.2,
                                return_sequences=True))(metafeatures_mask)

                metafeatures_LSTM_value = Bidirectional(LSTM(num_lstm_units, dropout=dropout, recurrent_dropout=0.2,
                                return_sequences=True))(metafeatures_mask)
        if use_embeddings:
            emb_LSTM = AdditiveAttention(name = 'Attention_Embeddings')([emb_LSTM_query, emb_LSTM_value])
        if use_metafeatures:
            metafeatures_LSTM = AdditiveAttention(name = 'Attention_Metafeatures')([metafeatures_LSTM_query, metafeatures_LSTM_value])
    else:
        if use_embeddings:
            emb_LSTM =  Bidirectional(LSTM(num_lstm_units, dropout=dropout, recurrent_dropout=dropout,
                        return_sequences=True))(emb_mask)
        if use_metafeatures:
            metafeatures_LSTM =  Bidirectional(LSTM(num_lstm_units, dropout=dropout, recurrent_dropout=dropout,
                            return_sequences=True))(metafeatures_mask)
    
    if use_embeddings and use_metafeatures:
    # Concatenating the two inputs
        model = Concatenate()([emb_LSTM, metafeatures_LSTM])
    elif use_metafeatures:
        model = metafeatures_LSTM

    
    # Adding attention and another LSTM to the concatenated layers
    if attention == 1:
        model_query =  Bidirectional(LSTM(num_lstm_units, dropout=dropout, recurrent_dropout=0.2, return_sequences=False))(model)
        model_value =  Bidirectional(LSTM(num_lstm_units, dropout=dropout, recurrent_dropout=0.2, return_sequences=False))(model)
        model = AdditiveAttention(name = 'Attention_Model')([model_query, model_value])

    else:
        model =  Bidirectional(LSTM(num_lstm_units, dropout=dropout, recurrent_dropout=dropout, return_sequences=False))(model)    

    # Adding dense layer with varying layers and units using parameter search
    for nl in range(num_dense_layers):
        model = Dense(num_dense_units)(model)
        model = LeakyReLU()(model)

    # Adding dropout to the model
    model = Dropout(dropout)(model)

    # Adding softmax dense layer with varying l2 regularizers using parameter search
    output = Dense(3, activation='softmax',
                    activity_regularizer=regularizers.l2(l2reg),
                    name = 'labels')(model)

    # Model output
    if use_embeddings and use_metafeatures:
        model = Model(inputs=[emb_input, metafeatures_input], outputs=output)
    elif use_metafeatures:
        model = Model(inputs=metafeatures_input, outputs=output)
    #model = Model(inputs=emb_input, outputs=output)
    # Plotting the model 
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # Adding Adam optimizer with varying learning rate using parameter search
    adam = optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999,
                           epsilon=1e-08, decay=0.0)


    # Compiling model
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    callback_list = []
    #TensorBoard
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    callback_list.append(tensorboard_callback)
    
    #Early_Stopping
    if Early_Stopping:
        earlystop_callback = EarlyStopping(monitor='val_accuracy', min_delta=0.0001, patience=5)
        callback_list.append(earlystop_callback)

    #plot_model(model, "model.png") 
    if Early_Stopping:
        # Fitting the model with varying batch sizes and epochs using parameter search        
        if use_embeddings and use_metafeatures:
            model.fit({'Embeddings': x_train_embeddings, 'Metafeatures': x_train_metafeatures}, y_train,
                    batch_size=mb_size,
                    epochs=num_epochs, shuffle=True, class_weight=None, verbose=1, callbacks=callback_list, validation_split=.1)
        elif use_metafeatures:
            model.fit(x_train_metafeatures, y_train,
                    batch_size=mb_size,
                    epochs=num_epochs, shuffle=True, class_weight=None, verbose=1, callbacks=callback_list, validation_split=.1)
    else:
         # Fitting the model with varying batch sizes and epochs using parameter search        
        if use_embeddings and use_metafeatures:
            model.fit({'Embeddings': x_train_embeddings, 'Metafeatures': x_train_metafeatures}, y_train,
                    batch_size=mb_size,
                    epochs=num_epochs, shuffle=True, class_weight=None, verbose=1, callbacks=callback_list)
        elif use_metafeatures:
            model.fit(x_train_metafeatures, y_train,
                    batch_size=mb_size,
                    epochs=num_epochs, shuffle=True, class_weight=None, verbose=1, callbacks=callback_list)
        

    # Evaluation time
    if eval==True:

        model.save('output\\model_veracity.h5')
        json_string = model.to_json()
        with open('output\\model_architecture_veracity.json','w') as fout:
            json.dump(json_string,fout)
        model.save_weights('output\\model_veracity_weights.h5')

    # Getting confidence of the model
    if use_embeddings and use_metafeatures:
        pred_probabilities = model.predict([x_test_embeddings, x_test_metafeatures], batch_size=mb_size, verbose=0)
        confidence = np.max(pred_probabilities, axis=1)

        # Getting predictions of the model
        y_prob = model.predict([x_test_embeddings, x_test_metafeatures], batch_size=mb_size)
        Y_pred = y_prob.argmax(axis=-1)
    elif use_metafeatures:
        pred_probabilities = model.predict(x_test_metafeatures, batch_size=mb_size, verbose=0)
        confidence = np.max(pred_probabilities, axis=1)

        # Getting predictions of the model
        y_prob = model.predict(x_test_metafeatures, batch_size=mb_size)
        Y_pred = y_prob.argmax(axis=-1)

    return Y_pred, confidence


"""    # %%
def get_model_plot():
    emb_shape = (14, 768)
    metafeatures_shape = (14, 12)
    # Creating the two inputs
    emb_input = Input(shape = emb_shape)
    metafeatures_input = Input(shape = metafeatures_shape)

    # Adding masks to account for zero paddings
    emb_mask = (Masking(mask_value=0., input_shape=(None, emb_shape)))(emb_input)
    metafeatures_mask = (Masking(mask_value=0., input_shape=(None, metafeatures_shape)))(metafeatures_input)

    emb_LSTM = (LSTM(128, dropout=0.2, recurrent_dropout=0.2,
                        return_sequences=True))(emb_mask)
    metafeatures_LSTM = (LSTM(128, dropout=0.2, recurrent_dropout=0.2,
                        return_sequences=True))(metafeatures_mask)

    model = Concatenate()([emb_LSTM, metafeatures_LSTM])

    model = (LSTM(128, dropout=0.2, recurrent_dropout=0.2,
                        return_sequences=False))(model)
    
    model = Dense(10, activation='relu')(model)
    model = Dropout(0.5)(model)
    output = Dense(3, activation='softmax')(model)
    model = Model(inputs=[emb_input, metafeatures_input], outputs=output)


    return plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


# %%
get_model_plot()

# %%
"""