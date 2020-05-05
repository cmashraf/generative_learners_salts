import os, shutil
import numpy as np
from numpy import array
from numpy import argmax
from random import shuffle
import pandas as pd
import random

#Keras build
import keras
from keras import objectives
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Multiply, Add
from keras.optimizers import Adam, Nadam
from keras import backend as K
from keras.objectives import binary_crossentropy #objs or losses
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Layer
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D

#cation data
cations = pd.read_csv('../data/cations.csv')
cations = cations['smiles_string']
salts = pd.read_csv('../data/salts.csv')
salts = salts['smiles_string']
anions = pd.read_csv('../data/anions.csv')
anions = anions['smiles']

#supporting functions
import sys
sys.path.insert(0, '../')
from scripts import TwoMoleculeVAE


#setup model
import json
f = open("../data/salt_char_to_index.json","r")
char_to_index = json.loads(f.read())
char_set = set(char_to_index.keys())
char_list = list(char_to_index.keys())
chars_in_dict = len(char_list)

class latent_viewer(keras.callbacks.Callback):
    """
    custom callback to save vae models at 10 and 30 epochs
    """
    #supporting functions
    import sys
    import numpy as np
    import json
    sys.path.insert(0, '../')
    from scripts import TwoMoleculeVAE
    
    def on_epoch_end(self, epoch, logs={}):
        ### save model, will overwrite
        self.model.save("development/temp_{}_{}.h5".format(self.model.name, epoch))
        
#         ### prepare to load model
#         f = open("../data/salt_char_to_index.json","r")
#         char_to_index = json.loads(f.read())
#         char_set = set(char_to_index.keys())
#         vae = TwoMoleculeVAE()
#         vae.create(char_set, char_set, qspr=True, 
#                    weights_file='temp_{}.h5'.format(self.model.name),
#                    qspr_outputs=1)
        
#         #test the predictions
#         x_train_cat = np.load('../data/{}_x_train_cat.npy'.format(self.model.name))
#         z = vae.cation_encoder.predict(x_train_cat)
#         z = np.array(z)
#         np.save('../data/latent_arrays/{}_{}.npy'.format(self.model.name, epoch), z)
        
#         #remove the old h5 file
#         folder = './'
#         for the_file in os.listdir(folder):
#             file_path = os.path.join(folder, the_file)
#             try: 
#                 if os.path.isfile(file_path):
#                     if 'temp' in the_file:
#                         os.unlink(file_path)
#             except Exception as e:
#                 print(e)

#######################
### SINGLE PROPERTY ###
#######################
properties = ['thermal_conductivity', 'cpt', 'density', 'viscosity']
epochs = 50
for prop in properties:
    gen3vae = TwoMoleculeVAE()
    gen3vae.create(char_set, char_set, qspr=True, 
                   weights_file='../models/gen3_2mol_1mil_GDB17_mix_pure_5.h5',
                   qspr_outputs=1)
    
    gen3vae.autoencoder.name = prop
    
    x_train_cat = np.load('../data/{}_x_train_cat.npy'.format(prop))
    x_train_ani = np.load('../data/{}_x_train_ani.npy'.format(prop))
    x_test_cat = np.load('../data/{}_x_test_cat.npy'.format(prop))
    x_test_ani = np.load('../data/{}_x_test_ani.npy'.format(prop))
    y_train = np.load('../data/{}_y_train.npy'.format(prop))
    y_test = np.load('../data/{}_y_test.npy'.format(prop))
    # view latent space development
    saver = latent_viewer()
    history = gen3vae.autoencoder.fit([x_train_cat, x_train_ani], [x_train_cat, x_train_ani, y_train],
                          shuffle=False,
                          validation_data=([x_test_cat, x_test_ani], [x_test_cat, x_test_ani, y_test]),
                          epochs=epochs,
                          callbacks=[saver])
    with open('../models/history_latent_{}_{}.json'.format(gen3vae.autoencoder.name,epochs), 'w') as f:
         json.dump(history.history, f)



