import numpy as np
import keras
from keras import objectives
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Multiply, Add
from keras.optimizers import Adam, Nadam

from numpy import array
from numpy import argmax
from random import shuffle
import pandas as pd
import random

#Keras build
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
categories = pd.read_csv('../data/categories.csv')
categories = categories['category']
coldic = pd.read_csv('../data/coldic.csv')
coldic = coldic.to_dict(orient='records')[0]
salt_coldic = pd.read_csv('../data/salt_coldic.csv')
salt_coldic = salt_coldic.to_dict(orient='records')[0]
salt_categories = pd.read_csv('../data/salt_categories.csv')
salt_categories = salt_categories['category']
density_coldic = pd.read_csv('../data/density_coldic.csv')
density_coldic = density_coldic.to_dict(orient='records')[0]
density_categories = pd.read_csv('../data/density_categories.csv')
density_categories = density_categories['category']

#supporting functions
import sys
sys.path.insert(0, '../')
from scripts import *

#training array info
smile_max_length = 51
import json
f = open("../data/salt_char_to_index.json","r")
char_to_index = json.loads(f.read())
char_set = set(char_to_index.keys())
char_list = list(char_to_index.keys())
chars_in_dict = len(char_list)

#training data
chemvae = MoleculeVAE()
chemvae.create(char_set, max_length=smile_max_length)
df = pd.read_csv('../data/GDB/GDB17.1000000', names=['smiles'])
data_size = 100000

histories = []
for p in range(3):
    values = df['smiles'][data_size*p:data_size*(p+1)]
    padded_smiles =  [pad_smiles(i, smile_max_length) for i in values if pad_smiles(i, smile_max_length)]
    X_train = np.zeros((data_size, smile_max_length, chars_in_dict), dtype=np.float32)
    
    # for each i, randomly select whether to sample from GDB or cations (padded_smiles_2)
    for i, smile in enumerate(padded_smiles):
#    for i in range(data_size):
#         linearly_scaled_prob = random.random() < 0.5#i/data_size
#         if linearly_scaled_prob:
#             smile = random.choice(cations)
#        smile = random.choice(cations)
        for j, char in enumerate(smile):
            X_train[i, j, char_to_index[char]] = 1

    history = chemvae.autoencoder.fit(X_train, X_train, shuffle = False)
    histories.append(history.history)
   # chemvae.save('../models/1mil_GDB17_62smi_{}.h5'.format(p+1))