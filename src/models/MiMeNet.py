import os
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class MiMeNet():

    def __init__(self, input_len, output_len, num_layer=1, layer_nodes=128, 
                 l1=0.0001, l2=0.0001, dropout=0.25, batch_size=1024, patience=40,
                 lr=0.0001, seed=42, gaussian_noise=0):

        reg = tf.keras.regularizers.L1L2(l1, l2)
        
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.GaussianNoise(gaussian_noise))
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
    micro, metab = train
    best_corr = -1
    
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    r = 0
    
    for l in range(1,4):
        best_l_corr = -1
        best_lr = 0.0001
        best_l1 = 0.00001
        best_l2 = 0.00001
        best_layer_size = 32
        best_drop = 0.15
        
        is_better = True

        while is_better:
            
            p_list = []
            y_list = []
            
            for train_index, test_index in kf.split(micro):
                train_micro, test_micro = micro[train_index], micro[test_index]
                train_metab, test_metab = metab[train_index], metab[test_index]

                model = MiMeNet(train_micro.shape[1], train_metab.shape[1], l1=best_l1, l2=best_l2, dropout=best_drop,
                            num_layer=l, layer_nodes=best_layer_size, lr=best_lr)

                model.train((train_micro, train_metab))
                p = model.test((test_micro, test_metab))

                p_list = list(p_list) + list(p)
                y_list = list(y_list) + list(test_metab)

                model.destroy()
                tf.keras.backend.clear_session()
                
            corr = pd.DataFrame(np.array(y_list)).corrwith(pd.DataFrame(np.array(p_list)), method="spearman").mean()
            if corr >= best_l_corr:
                best_l_corr = corr
                best_layer_size = best_layer_size * 2
                is_better = True
                if corr > best_corr:
                    best_params = model.get_params()
                
            else:
                is_better = False
                best_layer_size = best_layer_size/2

        is_better = True
        

        while is_better:
            
            p_list = []
            y_list = []
            
            for train_index, test_index in kf.split(micro):
                train_micro, test_micro = micro[train_index], micro[test_index]
                train_metab, test_metab = metab[train_index], metab[test_index]

                model = MiMeNet(train_micro.shape[1], train_metab.shape[1], l1=best_l1, l2=best_l2, dropout=best_drop,
                            num_layer=l, layer_nodes=best_layer_size, lr=best_lr)

                model.train((train_micro, train_metab))
                p = model.test((test_micro, test_metab))

                p_list = list(p_list) + list(p)
                y_list = list(y_list) + list(test_metab)
                
                model.destroy()
                tf.keras.backend.clear_session()
                
            corr = pd.DataFrame(np.array(y_list)).corrwith(pd.DataFrame(np.array(p_list)), method="spearman").mean()
            if corr >= best_l_corr:
                best_l_corr = corr
                best_layer_size = best_lr * 2
                is_better = True
                if corr > best_corr:
                    best_params = model.get_params()
                
            else:
                is_better = False
                best_lr = best_lr/2

        is_better = True



        while is_better:
            
            p_list = []
            y_list = []
            
            for train_index, test_index in kf.split(micro):
                train_micro, test_micro = micro[train_index], micro[test_index]
                train_metab, test_metab = metab[train_index], metab[test_index]

                model = MiMeNet(train_micro.shape[1], train_metab.shape[1], l1=best_l1, l2=best_l2, dropout=best_drop,
                            num_layer=l, layer_nodes=best_layer_size, lr=best_lr)

                model.train((train_micro, train_metab))
                p = model.test((test_micro, test_metab))

                p_list = list(p_list) + list(p)
                y_list = list(y_list) + list(test_metab)

                model.destroy()
                tf.keras.backend.clear_session()
                                
            corr = pd.DataFrame(np.array(y_list)).corrwith(pd.DataFrame(np.array(p_list)), method="spearman").mean()
            if corr >= best_l_corr:
                best_l_corr = corr
                best_11 = best_l1 * 2
                is_better = True
                if corr > best_corr:
                    best_params = model.get_params()
                
            else:
                is_better = False
                best_l1 = best_l1/2

        is_better = True
        
        
        while is_better:
            
            p_list = []
            y_list = []
            
            for train_index, test_index in kf.split(micro):
                train_micro, test_micro = micro[train_index], micro[test_index]
                train_metab, test_metab = metab[train_index], metab[test_index]

                model = MiMeNet(train_micro.shape[1], train_metab.shape[1], l1=best_l1, l2=best_l2, dropout=best_drop,
                            num_layer=l, layer_nodes=best_layer_size, lr=best_lr)

                model.train((train_micro, train_metab))
                p = model.test((test_micro, test_metab))

                p_list = list(p_list) + list(p)
                y_list = list(y_list) + list(test_metab)

                model.destroy()
                tf.keras.backend.clear_session()
                                
            corr = pd.DataFrame(np.array(y_list)).corrwith(pd.DataFrame(np.array(p_list)), method="spearman").mean()
            if corr >= best_l_corr:
                best_l_corr = corr
                best_layer_size = best_l2 * 2
                is_better = True
                if corr > best_corr:
                    best_params = model.get_params()
                
            else:
                is_better = False
                best_l2 = best_l2/2

        is_better = True
        

        while is_better:
            
            p_list = []
            y_list = []
            
            for train_index, test_index in kf.split(micro):
                train_micro, test_micro = micro[train_index], micro[test_index]
                train_metab, test_metab = metab[train_index], metab[test_index]

                model = MiMeNet(train_micro.shape[1], train_metab.shape[1], l1=best_l1, l2=best_l2, dropout=best_drop,
                            num_layer=l, layer_nodes=best_layer_size, lr=best_lr)

                model.train((train_micro, train_metab))
                p = model.test((test_micro, test_metab))

                p_list = list(p_list) + list(p)
                y_list = list(y_list) + list(test_metab)
                
                model.destroy()
                tf.keras.backend.clear_session()
                                
            corr = pd.DataFrame(np.array(y_list)).corrwith(pd.DataFrame(np.array(p_list)), method="spearman").mean()
            if corr >= best_l_corr:
                best_l_corr = corr
                best_layer_size = best_drop * 2
                if corr > best_corr:
                    best_params = model.get_params()
                if best_drop > 1:
                    is_better = False
                else:
                    is_better = True
                
            else:
                is_better = False
                best_drop = best_drop/2

        is_better = True

    return best_params