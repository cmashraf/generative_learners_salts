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
from scripts import *


#setup model
f = open("../data/salt_char_to_index.json","r")
char_to_index = json.loads(f.read())
char_set = set(char_to_index.keys())
char_list = list(char_to_index.keys())
chars_in_dict = len(char_list)
properties = ['thermal_conductivity']

#####################
### DUAL PROPERTY ###
#####################

epochs = 100
for subset in rSubset(['cpt', 'density', 'viscosity', 'thermal_conductivity'], 2):
    properties = list(subset)
    prop = properties[0] + '_' + properties[1]
#    gen1vae = MoleculeVAE()
#    gen1vae.create(char_set, qspr=True, mol_inputs=2, weights_file='../models/gen1_2mol_1mil_GDB17_mix_pure_5.h5')
#    gen2vae = TwoMoleculeOneLatentVAE()
#    gen2vae.create(char_set, qspr=True, weights_file='../models/gen2_2mol_1mil_GDB17_mix_pure_5.h5')
    gen3vae = TwoMoleculeVAE()
    gen3vae.create(char_set, char_set, qspr=True, weights_file='../models/gen3_2mol_1mil_GDB17_mix_pure_5.h5', qspr_outputs=2)
    gen3vae.autoencoder.name = 'gen3vae_' + prop
    
    x_train_cat = np.load('../data/{}_x_train_cat.npy'.format(prop))
    x_train_ani = np.load('../data/{}_x_train_ani.npy'.format(prop))
    x_test_cat = np.load('../data/{}_x_test_cat.npy'.format(prop))
    x_test_ani = np.load('../data/{}_x_test_ani.npy'.format(prop))
    y_train = np.load('../data/{}_y_train.npy'.format(prop))
    y_test = np.load('../data/{}_y_test.npy'.format(prop))
    
#    history = gen1vae.autoencoder.fit([x_train_cat, x_train_ani], [x_train_cat, x_train_ani, y_train],
#                          shuffle=False,
#                          validation_data=([x_test_cat, x_test_ani], [x_test_cat, x_test_ani, y_test]),
#                          epochs=epochs)
#    gen1vae.save('../models/gen1vae_{}_{}.h5'.format(prop,epochs))
#    with open('../models/history_gen1vae_{}_{}.json'.format(prop,epochs), 'w') as f:
#        json.dump(history.history, f)
#        
#    history = gen2vae.autoencoder.fit([x_train_cat, x_train_ani], [x_train_cat, x_train_ani, y_train],
#                          shuffle=False,
#                          validation_data=([x_test_cat, x_test_ani], [x_test_cat, x_test_ani, y_test]),
#                          epochs=epochs)
#    gen2vae.save('../models/gen2vae_{}_{}.h5'.format(prop,epochs))
#    with open('../models/history_gen2vae_{}_{}.json'.format(prop,epochs), 'w') as f:
#        json.dump(history.history, f)
    saver = VAESaver()
    history = gen3vae.autoencoder.fit([x_train_cat, x_train_ani], [x_train_cat, x_train_ani, y_train],
                          shuffle=False,
                          validation_data=([x_test_cat, x_test_ani], [x_test_cat, x_test_ani, y_test]),
                          epochs=epochs,
                          callbacks=[saver])
    gen3vae.save('../models/{}_{}.h5'.format(gen3vae.autoencoder.name,epochs))
    with open('../models/history_{}_{}.json'.format(gen3vae.autoencoder.name,epochs), 'w') as f:
         json.dump(history.history, f)


#######################
# ### SINGLE PROPERTY ###
# #######################
# # epochs = 1
# # for prop in properties:
# #     gen1vae = MoleculeVAE()
# #     gen1vae.create(char_set, qspr=True, mol_inputs=2, weights_file='../models/gen1_2mol_1mil_GDB17_mix_pure_5.h5')
# #     gen2vae = TwoMoleculeOneLatentVAE()
# #     gen2vae.create(char_set, qspr=True, weights_file='../models/gen2_2mol_1mil_GDB17_mix_pure_5.h5')
# #     gen3vae = TwoMoleculeVAE()
# #     gen3vae.create(char_set, char_set, qspr=True, weights_file='../models/gen3_2mol_1mil_GDB17_mix_pure_5.h5')

# #     x_train_cat = np.load('../data/{}_x_train_cat.npy'.format(prop))
# #     x_train_ani = np.load('../data/{}_x_train_ani.npy'.format(prop))
# #     x_test_cat = np.load('../data/{}_x_test_cat.npy'.format(prop))
# #     x_test_ani = np.load('../data/{}_x_test_ani.npy'.format(prop))
# #     y_train = np.load('../data/{}_y_train.npy'.format(prop))
# #     y_test = np.load('../data/{}_y_test.npy'.format(prop))
    
# #     history = gen1vae.autoencoder.fit([x_train_cat, x_train_ani], [x_train_cat, x_train_ani, y_train],
# #                           shuffle=False,
# #                           validation_data=([x_test_cat, x_test_ani], [x_test_cat, x_test_ani, y_test]),
# #                           epochs=epochs)
# #     gen1vae.save('../models/gen1vae_{}_{}.h5'.format(prop,epochs))
# #     with open('../models/history_gen1vae_{}_{}.json'.format(prop,epochs), 'w') as f:
# #         json.dump(history.history, f)
        
# #     history = gen2vae.autoencoder.fit([x_train_cat, x_train_ani], [x_train_cat, x_train_ani, y_train],
# #                           shuffle=False,
# #                           validation_data=([x_test_cat, x_test_ani], [x_test_cat, x_test_ani, y_test]),
# #                           epochs=epochs)
# #     gen2vae.save('../models/gen2vae_{}_{}.h5'.format(prop,epochs))
# #     with open('../models/history_gen2vae_{}_{}.json'.format(prop,epochs), 'w') as f:
# #         json.dump(history.history, f)
        
# #     history = gen3vae.autoencoder.fit([x_train_cat, x_train_ani], [x_train_cat, x_train_ani, y_train],
# #                           shuffle=False,
# #                           validation_data=([x_test_cat, x_test_ani], [x_test_cat, x_test_ani, y_test]),
# #                           epochs=epochs)
# #     gen3vae.save('../models/gen3vae_{}_{}.h5'.format(prop,epochs))
# #     with open('../models/history_gen3vae_{}_{}.json'.format(prop,epochs), 'w') as f:
# #         json.dump(history.history, f)

# epochs = 10
# for prop in properties:
# #    gen1vae = MoleculeVAE()
# #    gen1vae.create(char_set, qspr=True, mol_inputs=2, weights_file='../models/gen1_2mol_1mil_GDB17_mix_pure_5.h5')
# #    gen2vae = TwoMoleculeOneLatentVAE()
# #    gen2vae.create(char_set, qspr=True, weights_file='../models/gen2_2mol_1mil_GDB17_mix_pure_5.h5')
#    gen3vae = TwoMoleculeVAE()
#    gen3vae.create(char_set, char_set, qspr=True, weights_file='../models/gen3_2mol_1mil_GDB17_mix_pure_5.h5')
#    x_train_cat = np.load('../data/{}_x_train_cat.npy'.format(prop))
#    x_train_ani = np.load('../data/{}_x_train_ani.npy'.format(prop))
#    x_test_cat = np.load('../data/{}_x_test_cat.npy'.format(prop))
#    x_test_ani = np.load('../data/{}_x_test_ani.npy'.format(prop))
#    y_train = np.load('../data/{}_y_train.npy'.format(prop))
#    y_test = np.load('../data/{}_y_test.npy'.format(prop))
   
# #    history = gen1vae.autoencoder.fit([x_train_cat, x_train_ani], [x_train_cat, x_train_ani, y_train],
# #                          shuffle=False,
# #                          validation_data=([x_test_cat, x_test_ani], [x_test_cat, x_test_ani, y_test]),
# #                          epochs=epochs)
# #    gen1vae.save('../models/gen1vae_{}_{}.h5'.format(prop,epochs))
# #    with open('../models/history_gen1vae_{}_{}.json'.format(prop,epochs), 'w') as f:
# #        json.dump(history.history, f)
       
# #    history = gen2vae.autoencoder.fit([x_train_cat, x_train_ani], [x_train_cat, x_train_ani, y_train],
# #                          shuffle=False,
# #                          validation_data=([x_test_cat, x_test_ani], [x_test_cat, x_test_ani, y_test]),
# #                          epochs=epochs)
# #    gen2vae.save('../models/gen2vae_{}_{}.h5'.format(prop,epochs))
# #    with open('../models/history_gen2vae_{}_{}.json'.format(prop,epochs), 'w') as f:
# #        json.dump(history.history, f)
       
#    history = gen3vae.autoencoder.fit([x_train_cat, x_train_ani], [x_train_cat, x_train_ani, y_train],
#                          shuffle=False,
#                          validation_data=([x_test_cat, x_test_ani], [x_test_cat, x_test_ani, y_test]),
#                          epochs=epochs)
#    gen3vae.save('../models/gen3vae_{}_{}.h5'.format(prop,epochs))
#    with open('../models/history_gen3vae_{}_{}.json'.format(prop,epochs), 'w') as f:
#        json.dump(history.history, f)
        
# epochs = 30
# for prop in properties:
#     #gen1vae = MoleculeVAE()
#     #gen1vae.create(char_set, qspr=True, mol_inputs=2, weights_file='../models/gen1_2mol_1mil_GDB17_mix_pure_5.h5')
#     #gen2vae = TwoMoleculeOneLatentVAE()
#     #gen2vae.create(char_set, qspr=True, weights_file='../models/gen2_2mol_1mil_GDB17_mix_pure_5.h5')
#     gen3vae = TwoMoleculeVAE()
#     gen3vae.create(char_set, char_set, qspr=True, weights_file='../models/gen3_2mol_1mil_GDB17_mix_pure_5.h5')
#     x_train_cat = np.load('../data/{}_x_train_cat.npy'.format(prop))
#     x_train_ani = np.load('../data/{}_x_train_ani.npy'.format(prop))
#     x_test_cat = np.load('../data/{}_x_test_cat.npy'.format(prop))
#     x_test_ani = np.load('../data/{}_x_test_ani.npy'.format(prop))
#     y_train = np.load('../data/{}_y_train.npy'.format(prop))
#     y_test = np.load('../data/{}_y_test.npy'.format(prop))
    
#     #history = gen1vae.autoencoder.fit([x_train_cat, x_train_ani], [x_train_cat, x_train_ani, y_train],
#     #                      shuffle=False,
#     #                      validation_data=([x_test_cat, x_test_ani], [x_test_cat, x_test_ani, y_test]),
#     #                      epochs=epochs)
#     #gen1vae.save('../models/gen1vae_{}_{}.h5'.format(prop,epochs))
#     #with open('../models/history_gen1vae_{}_{}.json'.format(prop,epochs), 'w') as f:
#     #    json.dump(history.history, f)
#     #    
#     #history = gen2vae.autoencoder.fit([x_train_cat, x_train_ani], [x_train_cat, x_train_ani, y_train],
#     #                      shuffle=False,
#     #                      validation_data=([x_test_cat, x_test_ani], [x_test_cat, x_test_ani, y_test]),
#     #                      epochs=epochs)
#     #gen2vae.save('../models/gen2vae_{}_{}.h5'.format(prop,epochs))
#     #with open('../models/history_gen2vae_{}_{}.json'.format(prop,epochs), 'w') as f:
#     #    json.dump(history.history, f)
        
#     history = gen3vae.autoencoder.fit([x_train_cat, x_train_ani], [x_train_cat, x_train_ani, y_train],
#                           shuffle=False,
#                           validation_data=([x_test_cat, x_test_ani], [x_test_cat, x_test_ani, y_test]),
#                           epochs=epochs)
#     gen3vae.save('../models/gen3vae_{}_{}.h5'.format(prop,epochs))
#     with open('../models/history_gen3vae_{}_{}.json'.format(prop,epochs), 'w') as f:
#         json.dump(history.history, f)

# epochs = 100
# for prop in properties:
# #    gen1vae = MoleculeVAE()
# #    gen1vae.create(char_set, qspr=True, mol_inputs=2, weights_file='../models/gen1_2mol_1mil_GDB17_mix_pure_5.h5')
# #    gen2vae = TwoMoleculeOneLatentVAE()
# #    gen2vae.create(char_set, qspr=True, weights_file='../models/gen2_2mol_1mil_GDB17_mix_pure_5.h5')
#     gen3vae = TwoMoleculeVAE()
#     gen3vae.create(char_set, char_set, qspr=True, weights_file='../models/gen3_2mol_1mil_GDB17_mix_pure_5.h5')
#     x_train_cat = np.load('../data/{}_x_train_cat.npy'.format(prop))
#     x_train_ani = np.load('../data/{}_x_train_ani.npy'.format(prop))
#     x_test_cat = np.load('../data/{}_x_test_cat.npy'.format(prop))
#     x_test_ani = np.load('../data/{}_x_test_ani.npy'.format(prop))
#     y_train = np.load('../data/{}_y_train.npy'.format(prop))
#     y_test = np.load('../data/{}_y_test.npy'.format(prop))
    
# #    history = gen1vae.autoencoder.fit([x_train_cat, x_train_ani], [x_train_cat, x_train_ani, y_train],
# #                          shuffle=False,
# #                          validation_data=([x_test_cat, x_test_ani], [x_test_cat, x_test_ani, y_test]),
# #                          epochs=epochs)
# #    gen1vae.save('../models/gen1vae_{}_{}.h5'.format(prop,epochs))
# #    with open('../models/history_gen1vae_{}_{}.json'.format(prop,epochs), 'w') as f:
# #        json.dump(history.history, f)
# #        
# #    history = gen2vae.autoencoder.fit([x_train_cat, x_train_ani], [x_train_cat, x_train_ani, y_train],
# #                          shuffle=False,
# #                          validation_data=([x_test_cat, x_test_ani], [x_test_cat, x_test_ani, y_test]),
# #                          epochs=epochs)
# #    gen2vae.save('../models/gen2vae_{}_{}.h5'.format(prop,epochs))
# #    with open('../models/history_gen2vae_{}_{}.json'.format(prop,epochs), 'w') as f:
# #        json.dump(history.history, f)
        
#     history = gen3vae.autoencoder.fit([x_train_cat, x_train_ani], [x_train_cat, x_train_ani, y_train],
#                           shuffle=False,
#                           validation_data=([x_test_cat, x_test_ani], [x_test_cat, x_test_ani, y_test]),
#                           epochs=epochs)
#     gen3vae.save('../models/gen3vae_{}_{}.h5'.format(prop,epochs))
#     with open('../models/history_gen3vae_{}_{}.json'.format(prop,epochs), 'w') as f:
#         json.dump(history.history, f)
