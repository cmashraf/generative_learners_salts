import matplotlib.pylab as plt
import numpy as np
import seaborn as sns; sns.set()

import keras
from keras.models import Sequential, Model
from keras.layers import Dense
from keras.optimizers import Adam
import salty
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from random import shuffle
import pandas as pd
import random

#Keras build
from keras import backend as K
from keras.objectives import binary_crossentropy #objs or losses
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D

#cation data
cations = pd.read_csv('../data/cations.csv')
cations = cations['smiles_string']
categories = pd.read_csv('../data/categories.csv')
categories = categories['category']
coldic = pd.read_csv('../data/coldic.csv')
coldic = coldic.to_dict(orient='records')[0]

#supporting functions
import sys
sys.path.insert(0, '../')
from scripts import *

#For loop to save semi-trained model to
#view PCAs and Z distributions during training

#training array info
smile_max_length = 51
import json
f = open("../data/gdb_char_to_index.json","r")
char_to_index = json.loads(f.read())
char_set = set(char_to_index.keys())
char_list = list(char_to_index.keys())
chars_in_dict = len(char_list)

#training data
df = pd.read_csv('../data/GDB/GDB17.1000000', names=['smiles'])
data_size = 100000
chemvae = MoleculeVAE()
chemvae.create(char_set, 51)

for p in range(10):
    values = df['smiles'][data_size*p:data_size*(p+1)]
    padded_smiles =  [pad_smiles(i, smile_max_length) for i in values if pad_smiles(i, smile_max_length)]
    X_train = np.zeros((data_size, smile_max_length, chars_in_dict), dtype=np.float32)

    for i, smile in enumerate(padded_smiles):
        for j, char in enumerate(smile):
            X_train[i, j, char_to_index[char]] = 1
    chemvae.autoencoder.fit(X_train, X_train, shuffle = False)
#     chemvae.save('1mil_GDB17_{}.h5'.format(p+1))
    
