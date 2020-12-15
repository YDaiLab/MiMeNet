import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import spearmanr
from sklearn.metrics import make_scorer
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class MiMeNet():

    def __init__(self, input_len, output_len, num_layer=1, layer_nodes=128, 
                 l1=0.0001, l2=0.0001, dropout=0.25, batch_size=1024, patience=40,
                 lr=0.0001, seed=42, gaussian_noise=0):

        reg = tf.keras.regularizers.L1L2(l1, l2)
        
        self.model = tf.keras.Sequential()
        for l in range(num_layer):
            self.model.add(tf.keras.layers.Dense(layer_nodes, activation='relu', kernel_regularizer=reg, bias_regularizer=reg, name="fc" + str(l)))
            self.model.add(tf.keras.layers.Dropout(dropout))
        self.model.add(tf.keras.layers.Dense(output_len, activation='linear', kernel_regularizer=reg, bias_regularizer=reg, name="output"))

        self.num_layer = num_layer
        self.layer_nodes = layer_nodes
        self.l1 = l1
        self.l2 = l2
        self.dropout = dropout
        self.learning_rate = lr
        
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed

    def train(self, train):
        train_x, train_y = train
        num_class = train_y.shape[1]

        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss='MSE')
        es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=self.patience, restore_best_weights=True)

        self.model.fit(train_x, train_y, batch_size=self.batch_size, verbose=0, epochs=100000, 
                       callbacks=[es_cb], validation_split=0.2)
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

    def get_params(self):
        return self.num_layer, self.layer_nodes, self.l1, self.l2, self.dropout, self.learning_rate
    
def tune_MiMeNet(train, seed=None):
    best_score = -1000
    best_params = []
    
    micro, metab = train    
    def build_model(num_layer=1, layer_nodes=128, 
                 l1=0.0001, l2=0.0001, dropout=0.25, batch_size=1024, patience=40,
                 lr=0.0001, seed=42, gaussian_noise=0):

        reg = tf.keras.regularizers.L1L2(l1, l2)
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(micro.shape[1]))
        for l in range(num_layer):
            model.add(tf.keras.layers.Dense(layer_nodes, activation='relu', 
                                                 kernel_regularizer=reg, bias_regularizer=reg, name="fc" + str(l)))
            model.add(tf.keras.layers.Dropout(dropout))
        model.add(tf.keras.layers.Dense(metab.shape[1], activation='linear', 
                                             kernel_regularizer=reg, bias_regularizer=reg, name="output"))
        model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='MSE', metrics=["mse"])
        return model
    es_cb = tf.keras.callbacks.EarlyStopping('val_loss', patience=40, restore_best_weights=True)

    def my_custom_loss_func(y_true, y_pred):
        diff = np.float(pd.DataFrame(y_true).corrwith(pd.DataFrame(y_pred)).mean())
        return diff
    
    scorer = make_scorer(my_custom_loss_func, greater_is_better=True)
    callbacks = [es_cb]
    
    #_l1_lambda = np.logspace(-4,-2,50)
    _l1_lambda = [0]
    _l2_lambda = np.logspace(-4,-1,10)
    _num_layer = [1,2,3]
    _layer_nodes = [32,128,512]
    _dropout = [0.1,0.3,0.5]
    _learning_rate = [0.001]
    params=dict(l1=_l1_lambda, l2=_l2_lambda, num_layer=_num_layer,
                layer_nodes = _layer_nodes, dropout=_dropout, lr=_learning_rate)
    

    
    
    for i in range(20):
        model = KerasRegressor(build_fn=build_model,epochs=1000,batch_size=1024, verbose=0)

        rscv = RandomizedSearchCV(model,param_distributions=params, cv=5, n_iter=1, 
                              scoring=scorer)
        rscv_results = rscv.fit(micro,metab,callbacks=callbacks, validation_split=0.2)
        if rscv_results.best_score_ > best_score:
            best_score = rscv_results.best_score_
            best_params = rscv_results.best_params_
            tf.keras.backend.clear_session()
    
    print('Best score is: {} using {}'.format(best_score, best_params))
    return best_params