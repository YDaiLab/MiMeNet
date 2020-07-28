import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class MLPNN():

    def __init__(self, input_len, output_len, num_layer=1, layer_nodes=128, 
                 l1=0.0001, l2=0.0001, dropout=0.25, batch_size=1024, patience=40,
                 lr=0.0001, seed=42, gaussian_noise=0):

        reg = tf.keras.regularizers.L1L2(l1, l2)
        
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.GaussianNoise(gaussian_noise))
        for l in range(num_layer):
            self.model.add(tf.keras.layers.Dense(layer_nodes, activation='relu', kernel_regularizer=reg, bias_regularizer=reg, name="fc" + str(l)))
            self.model.add(tf.keras.layers.Dropout(dropout))
        self.model.add(tf.keras.layers.Dense(output_len, activation='softmax', kernel_regularizer=reg, bias_regularizer=reg, name="output"))

        self.patience = patience
        self.learning_rate = lr
        self.batch_size = batch_size
        self.seed = seed

    def train(self, train, validation=[], train_weights=[]):
        train_x, train_y = train
        num_class = train_y.shape[1]

        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss='categorical_crossentropy')
        es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=self.patience, restore_best_weights=True)

        self.model.fit(train_x, train_y, batch_size=self.batch_size, verbose=0, epochs=100000, 
                       callbacks=[es_cb], validation_split=0.1)
        self.model.fit(train_x, train_y, batch_size=self.batch_size, verbose=0, epochs=1)
        return

    def test(self, test):
        test_x, test_y = test
        num_class = test_y.shape[1]
        preds = self.model.predict(test_x)
        return preds

    def get_scores(self):
        w_list = []
        for l in self.model.layers:
            if len(l.get_weights()) > 0:
                if l.get_weights()[0].ndim == 2:
                    w_list.append(l.get_weights()[0])
        num_layers = len(w_list)
        scores = w_list[0]
        for w in range(1,num_layers):
            scores = np.matmul(scores, w_list[w])
        return scores

    def destroy(self):
        tf.keras.backend.clear_session()
        return
