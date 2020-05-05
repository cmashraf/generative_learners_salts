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

chemvae = MoleculeVAE()
chemvae.create(char_set, 
               mol_inputs=2,
               weights_file='../models/gen1_2mol_50mil_GDB17_1.h5')
data_size = 1000000
histories = []

############        
###STEP 1###
############
smile_max_length = 62
df1 = pd.read_csv('../data/GDB/GDB17_shuffled_1_25M.csv', names=['smiles'])
df2 = pd.read_csv('../data/GDB/GDB17_shuffled_2_25M.csv', names=['smiles'])
padded_cations =  [pad_smiles(i, smile_max_length) for i in cations if pad_smiles(i, smile_max_length)]
padded_anions =  [pad_smiles(i, smile_max_length) for i in anions if pad_smiles(i, smile_max_length)]

for p in range(0, 25):
    values1 = df1['smiles'][data_size*p:data_size*(p+1)]
    values2 = df2['smiles'][data_size*p:data_size*(p+1)]
    padded_smiles1 =  [pad_smiles(i, smile_max_length) for i in values1 if pad_smiles(i, smile_max_length)]
    padded_smiles2 =  [pad_smiles(i, smile_max_length) for i in values2 if pad_smiles(i, smile_max_length)]
    
    
    X_train_cat = np.zeros((data_size, smile_max_length, chars_in_dict), dtype=np.float32)
    X_train_ani = np.zeros((data_size, smile_max_length, chars_in_dict), dtype=np.float32)
    
    #for each i, randomly select whether to sample from GDB or cations (padded_smiles_2)
    anions_added = 0
    cations_added = 0
    for i, smile in enumerate(padded_smiles1):
#         linearly_scaled_prob = random.random() < i/data_size
#         if linearly_scaled_prob:
#             smile = random.choice(padded_anions)
#             anions_added += 1
        for j, char in enumerate(smile):
            X_train_cat[i, j, char_to_index[char]] = 1
    for i, smile in enumerate(padded_smiles2):
#         linearly_scaled_prob = random.random() < i/data_size
#         if linearly_scaled_prob:
#             smile = random.choice(padded_anions)
#             anions_added += 1
        for j, char in enumerate(smile):
            X_train_ani[i, j, char_to_index[char]] = 1
            
#     print('anions added: {}'.format(anions_added))

#     X_train, X_test = train_test_split(X_train, test_size=0.01, random_state=42)   
    history = chemvae.autoencoder.fit([X_train_cat, X_train_ani], [X_train_cat, X_train_ani],
                                      shuffle = False, 
                                      epochs=10)
    histories.append(history.history)
    name = 'gen1_2mol_50mil_GDB17_10epoch'
    chemvae.save('../models/{}_{}.h5'.format(name,p+1))    
    with open('../models/history_{}.json'.format(name), 'w') as f:
            json.dump(history.history, f)
    values1 = values2 = None
    padded_smiles1 = padded_smiles2 = None
    X_train_cat = X_train_ani = None
with open('../models/history_{}.json'.format(name), 'w') as f:
            json.dump(histories, f)
    
        
############     
###STEP 2###
############
#smile_max_length = 62
#df1 = pd.read_csv('../data/GDB/GDB17.3000000', names=['smiles'])
#df2 = pd.read_csv('../data/GDB/GDB17.4000000', names=['smiles'])
#
#for p in range(5):
#    values1 = df1['smiles'][data_size*p:data_size*(p+1)]
#    values2 = df2['smiles'][data_size*p:data_size*(p+1)]
#    padded_smiles1 =  [pad_smiles(i, smile_max_length) for i in values1 if pad_smiles(i, smile_max_length)]
#    padded_smiles2 =  [pad_smiles(i, smile_max_length) for i in values2 if pad_smiles(i, smile_max_length)]
#    padded_cations =  [pad_smiles(i, smile_max_length) for i in cations if pad_smiles(i, smile_max_length)]
#    padded_anions =  [pad_smiles(i, smile_max_length) for i in anions if pad_smiles(i, smile_max_length)]
#    
#    X_train_cat = np.zeros((data_size, smile_max_length, chars_in_dict), dtype=np.float32)
#    X_train_ani = np.zeros((data_size, smile_max_length, chars_in_dict), dtype=np.float32)
#    
#    #for each i, randomly select whether to sample from GDB or cations (padded_smiles_2)
#    anions_added = 0
#    cations_added = 0
#    for i, smile in enumerate(padded_smiles1):
#        linearly_scaled_prob = random.random() < i/data_size
#        if linearly_scaled_prob:
#            smile = random.choice(padded_cations)
#            cations_added += 1
#        for j, char in enumerate(smile):
#            X_train_cat[i, j, char_to_index[char]] = 1
#    for i, smile in enumerate(padded_smiles2):
#        linearly_scaled_prob = random.random() < i/data_size
#        if linearly_scaled_prob:
#            smile = random.choice(padded_anions)
#            anions_added += 1
#        for j, char in enumerate(smile):
#            X_train_ani[i, j, char_to_index[char]] = 1
#    print('cations added: {}'.format(cations_added))        
#    print('anions added: {}'.format(anions_added))
#
##     X_train, X_test = train_test_split(X_train, test_size=0.01, random_state=42)   
#    history = chemvae.autoencoder.fit([X_train_cat, X_train_ani], [X_train_cat, X_train_ani],
#                                      shuffle = False)
#    histories.append(history.history)
#    name = 'gen1_2mol_1mil_GDB17_mix'
#    chemvae.save('../models/{}_{}.h5'.format(name,p+1))
#    
#with open('../models/history_{}.json'.format(name), 'w') as f:
#        json.dump(histories, f)
#
#############        
####STEP 3###
#############
#for p in range(5):
#    values1 = df1['smiles'][data_size*p:data_size*(p+1)]
#    values2 = df2['smiles'][data_size*p:data_size*(p+1)]
#    padded_smiles1 =  [pad_smiles(i, smile_max_length) for i in values1 if pad_smiles(i, smile_max_length)]
#    padded_smiles2 =  [pad_smiles(i, smile_max_length) for i in values2 if pad_smiles(i, smile_max_length)]
#    padded_cations =  [pad_smiles(i, smile_max_length) for i in cations if pad_smiles(i, smile_max_length)]
#    padded_anions =  [pad_smiles(i, smile_max_length) for i in anions if pad_smiles(i, smile_max_length)]
#    
#    X_train_cat = np.zeros((data_size, smile_max_length, chars_in_dict), dtype=np.float32)
#    X_train_ani = np.zeros((data_size, smile_max_length, chars_in_dict), dtype=np.float32)
#    
#    #for each i, randomly select whether to sample from GDB or cations (padded_smiles_2)
#    anions_added = 0
#    cations_added = 0
#    for i, smile in enumerate(padded_smiles1):
#        smile = random.choice(padded_cations)
#        cations_added += 1
#        for j, char in enumerate(smile):
#            X_train_cat[i, j, char_to_index[char]] = 1
#    for i, smile in enumerate(padded_smiles2):
#        smile = random.choice(padded_anions)
#        anions_added += 1
#        for j, char in enumerate(smile):
#            X_train_ani[i, j, char_to_index[char]] = 1
#    print('cations added: {}'.format(cations_added))        
#    print('anions added: {}'.format(anions_added))
#    history = chemvae.autoencoder.fit([X_train_cat, X_train_ani], [X_train_cat, X_train_ani],
#                                      shuffle = False)
#    histories.append(history.history)
#    name = 'gen1_2mol_1mil_GDB17_mix_pure'
#    chemvae.save('../models/{}_{}.h5'.format(name,p+1))
#    
#with open('../models/history_{}.json'.format(name), 'w') as f:
#        json.dump(histories, f)
#        
#############
###EXTREEME#
############
chemvae = TwoMoleculeOneLatentVAE()
chemvae.create(char_set)
data_size = 1000000
histories = []

############        
###STEP 1###
############

for p in range(25):
    values1 = df1['smiles'][data_size*p:data_size*(p+1)]
    values2 = df2['smiles'][data_size*p:data_size*(p+1)]
    padded_smiles1 =  [pad_smiles(i, smile_max_length) for i in values1 if pad_smiles(i, smile_max_length)]
    padded_smiles2 =  [pad_smiles(i, smile_max_length) for i in values2 if pad_smiles(i, smile_max_length)]
    padded_cations =  [pad_smiles(i, smile_max_length) for i in cations if pad_smiles(i, smile_max_length)]
    padded_anions =  [pad_smiles(i, smile_max_length) for i in anions if pad_smiles(i, smile_max_length)]
    
    X_train_cat = np.zeros((data_size, smile_max_length, chars_in_dict), dtype=np.float32)
    X_train_ani = np.zeros((data_size, smile_max_length, chars_in_dict), dtype=np.float32)
    
    #for each i, randomly select whether to sample from GDB or cations (padded_smiles_2)
    anions_added = 0
    cations_added = 0
    for i, smile in enumerate(padded_smiles1):
#         linearly_scaled_prob = random.random() < i/data_size
#         if linearly_scaled_prob:
#             smile = random.choice(padded_anions)
#             anions_added += 1
        for j, char in enumerate(smile):
            X_train_cat[i, j, char_to_index[char]] = 1
    for i, smile in enumerate(padded_smiles2):
#         linearly_scaled_prob = random.random() < i/data_size
#         if linearly_scaled_prob:
#             smile = random.choice(padded_anions)
#             anions_added += 1
        for j, char in enumerate(smile):
            X_train_ani[i, j, char_to_index[char]] = 1
            
#     print('anions added: {}'.format(anions_added))

#     X_train, X_test = train_test_split(X_train, test_size=0.01, random_state=42)   
    history = chemvae.autoencoder.fit([X_train_cat, X_train_ani], [X_train_cat, X_train_ani],
                                      shuffle = False)
    histories.append(history.history)
    name = 'gen2_2mol_50mil_GDB17'
    chemvae.save('../models/{}_{}.h5'.format(name,p+1))
    with open('../models/history_{}.json'.format(name), 'w') as f:
        json.dump(history.history, f)

    
with open('../models/history_{}.json'.format(name), 'w') as f:
        json.dump(histories, f)
        
############     
###STEP 2###
############
smile_max_length = 62
df1 = pd.read_csv('../data/GDB/GDB17.3000000', names=['smiles'])
df2 = pd.read_csv('../data/GDB/GDB17.4000000', names=['smiles'])

for p in range(5):
    values1 = df1['smiles'][data_size*p:data_size*(p+1)]
    values2 = df2['smiles'][data_size*p:data_size*(p+1)]
    padded_smiles1 =  [pad_smiles(i, smile_max_length) for i in values1 if pad_smiles(i, smile_max_length)]
    padded_smiles2 =  [pad_smiles(i, smile_max_length) for i in values2 if pad_smiles(i, smile_max_length)]
    padded_cations =  [pad_smiles(i, smile_max_length) for i in cations if pad_smiles(i, smile_max_length)]
    padded_anions =  [pad_smiles(i, smile_max_length) for i in anions if pad_smiles(i, smile_max_length)]
    
    X_train_cat = np.zeros((data_size, smile_max_length, chars_in_dict), dtype=np.float32)
    X_train_ani = np.zeros((data_size, smile_max_length, chars_in_dict), dtype=np.float32)
    
    #for each i, randomly select whether to sample from GDB or cations (padded_smiles_2)
    anions_added = 0
    cations_added = 0
    for i, smile in enumerate(padded_smiles1):
        linearly_scaled_prob = random.random() < i/data_size
        if linearly_scaled_prob:
            smile = random.choice(padded_cations)
            cations_added += 1
        for j, char in enumerate(smile):
            X_train_cat[i, j, char_to_index[char]] = 1
    for i, smile in enumerate(padded_smiles2):
        linearly_scaled_prob = random.random() < i/data_size
        if linearly_scaled_prob:
            smile = random.choice(padded_anions)
            anions_added += 1
        for j, char in enumerate(smile):
            X_train_ani[i, j, char_to_index[char]] = 1
    print('cations added: {}'.format(cations_added))        
    print('anions added: {}'.format(anions_added))

#     X_train, X_test = train_test_split(X_train, test_size=0.01, random_state=42)   
    history = chemvae.autoencoder.fit([X_train_cat, X_train_ani], [X_train_cat, X_train_ani],
                                      shuffle = False)
    histories.append(history.history)
    name = 'gen2_2mol_1mil_GDB17_mix'
    chemvae.save('../models/{}_{}.h5'.format(name,p+1))
    
with open('../models/history_{}.json'.format(name), 'w') as f:
        json.dump(histories, f)

############        
###STEP 3###
############
for p in range(5):
    values1 = df1['smiles'][data_size*p:data_size*(p+1)]
    values2 = df2['smiles'][data_size*p:data_size*(p+1)]
    padded_smiles1 =  [pad_smiles(i, smile_max_length) for i in values1 if pad_smiles(i, smile_max_length)]
    padded_smiles2 =  [pad_smiles(i, smile_max_length) for i in values2 if pad_smiles(i, smile_max_length)]
    padded_cations =  [pad_smiles(i, smile_max_length) for i in cations if pad_smiles(i, smile_max_length)]
    padded_anions =  [pad_smiles(i, smile_max_length) for i in anions if pad_smiles(i, smile_max_length)]
    
    X_train_cat = np.zeros((data_size, smile_max_length, chars_in_dict), dtype=np.float32)
    X_train_ani = np.zeros((data_size, smile_max_length, chars_in_dict), dtype=np.float32)
    
    #for each i, randomly select whether to sample from GDB or cations (padded_smiles_2)
    anions_added = 0
    cations_added = 0
    for i, smile in enumerate(padded_smiles1):
        smile = random.choice(padded_cations)
        cations_added += 1
        for j, char in enumerate(smile):
            X_train_cat[i, j, char_to_index[char]] = 1
    for i, smile in enumerate(padded_smiles2):
        smile = random.choice(padded_anions)
        anions_added += 1
        for j, char in enumerate(smile):
            X_train_ani[i, j, char_to_index[char]] = 1
    print('cations added: {}'.format(cations_added))        
    print('anions added: {}'.format(anions_added))
    history = chemvae.autoencoder.fit([X_train_cat, X_train_ani], [X_train_cat, X_train_ani],
                                      shuffle = False)
    histories.append(history.history)
    name = 'gen2_2mol_1mil_GDB17_mix_pure'
    chemvae.save('../models/{}_{}.h5'.format(name,p+1))
    
with open('../models/history_{}.json'.format(name), 'w') as f:
        json.dump(histories, f)
