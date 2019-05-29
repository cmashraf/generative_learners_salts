
# coding: utf-8

# In[1]:


import numpy as np
import json

import salty
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import random

import keras
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
from keras.objectives import binary_crossentropy #objs or losses
from keras.layers import Input, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D


# In[2]:


def pad_smiles(smiles_string, smile_max_length):
     if len(smiles_string) < smile_max_length:
            return smiles_string + " " * (smile_max_length - len(smiles_string))
        
def one_hot(smi, char_to_index):
    test_smi = smi
    smile_max_length=51
    char_set = set(char_to_index.keys())
    test_smi = pad_smiles(test_smi, smile_max_length)
    Z = np.zeros((1, smile_max_length, len(list(char_set))), dtype=np.bool)
    for t, char in enumerate(test_smi):
        Z[0, t, char_to_index[char]] = 1
    return Z

def create_char_list(char_set, smile_series):
    for smile in smile_series:
        char_set.update(set(smile))
    return char_set


# In[10]:


## the following code generates 1mil GDB17 molecules from 50mil, shuffling them. 
#df = pd.read_csv('GDB17.50000000', names=['smiles'])
#from sklearn.utils import shuffle
#df = shuffle(df, random_state=1)
#first_mil = df[:1000000]
#first_mil.to_csv('GDB17.1000000')


# In[3]:


df = pd.read_csv('GDB17.1000000', names=['smiles'])


# In[4]:


df = df[df['smiles'].str.contains("N+", regex=False)]


# In[5]:


#create padded smiles of GDB17 molecules. 
values = df['smiles']
smile_max_length = values.map(len).max()
padded_smiles =  [pad_smiles(i, smile_max_length) for i in values if pad_smiles(i, smile_max_length)]


# In[6]:


#create char set of 1mil GDB17 molecules.
char_set = set()
char_set = create_char_list(char_set, padded_smiles)


# In[7]:


properties = ['density', 'cpt', 'viscosity', 'thermal_conductivity',
              'melting_point']
props = properties
devmodel = salty.aggregate_data(props, merge='Union')
devmodel.Data['smiles_string'] = devmodel.Data['smiles-cation']
cations = devmodel.Data['smiles_string'].drop_duplicates()
cations = cations.reset_index(drop=True)


# In[9]:


#create padded smiles of cations.
padded_smiles_2 =  [pad_smiles(i, smile_max_length) for i in cations if pad_smiles(i, smile_max_length)]


# In[10]:


#create char set of cations, update char set of GDB17 molecules to include this.
char_set_2 = set()
char_set_2 = create_char_list(char_set_2, padded_smiles_2)
char_set.update(set(char_set_2))


# In[11]:


char_list = list(char_set)
chars_in_dict = len(char_list)
char_to_index = dict((c, i) for i, c in enumerate(char_list))
index_to_char = dict((i, c) for i, c in enumerate(char_list))


# In[12]:


X_train1 = np.zeros((1000000, smile_max_length, chars_in_dict), dtype=np.float32)


# In[13]:


#first epoch, 1mil GDB.
for i, smile in enumerate(padded_smiles[:1000000]):
    for j, char in enumerate(smile):
        X_train1[i, j, char_to_index[char]] = 1


# In[14]:


#second epoch, 500K with 250K GDB, 250K cations.
X_train2 = np.zeros((500000, smile_max_length, chars_in_dict), dtype=np.float32)


# In[15]:


epoch2paddedsmiles = []

for i in range(250000):
    epoch2paddedsmiles.append(random.choice(padded_smiles_2))
    epoch2paddedsmiles.append(random.choice(padded_smiles))

random.shuffle(epoch2paddedsmiles)

for i, smile in enumerate(epoch2paddedsmiles[:500000]):
    for j, char in enumerate(smile):
        X_train2[i, j, char_to_index[char]] = 1


# In[16]:


#third epoch, just cations bootstrapped.
third_epoch_size = 500000
X_train3 = np.zeros((third_epoch_size, smile_max_length, chars_in_dict), dtype=np.float32)
for i in range(third_epoch_size):
    smile = random.choice(padded_smiles_2)
    for j, char in enumerate(smile):
        X_train3[i, j, char_to_index[char]] = 1


# In[17]:


X_train1, X_test1 = train_test_split(X_train1, test_size=0.33, random_state=42)
X_train2, X_test2 = train_test_split(X_train2, test_size=0.33, random_state=42)
X_train3, X_test3 = train_test_split(X_train3, test_size=0.33, random_state=42)


# In[18]:


from keras import backend as K
from keras.objectives import binary_crossentropy #objs or losses
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D


# In[19]:


def create_network(latent_rep_dim = 292):
    def Encoder(x, latent_rep_size, smile_max_length, epsilon_std = 0.01):
        h = Convolution1D(9, 9, activation = 'relu', name='conv_1')(x)
        h = Convolution1D(9, 9, activation = 'relu', name='conv_2')(h)
        h = Convolution1D(10, 11, activation = 'relu', name='conv_3')(h)
        h = Flatten(name = 'flatten_1')(h)
        h = Dense(435, activation = 'relu', name = 'dense_1')(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size),
                                      mean=0., stddev = epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = smile_max_length * binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) -                                      K.exp(z_log_var), axis = -1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,),
                                 name='lambda')([z_mean, z_log_var]))

    def Decoder(z, latent_rep_size, smile_max_length, charset_length):
        h = Dense(latent_rep_size, name='latent_input', activation = 'relu')(z)
        h = RepeatVector(smile_max_length, name='repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='gru_1')(h)
        h = GRU(501, return_sequences = True, name='gru_2')(h)
        h = GRU(501, return_sequences = True, name='gru_3')(h)
        return TimeDistributed(Dense(charset_length, activation='softmax'),
                               name='decoded_mean')(h)

    x = Input(shape=(smile_max_length, len(char_set)))
    _, z = Encoder(x, latent_rep_size=latent_rep_dim, smile_max_length=smile_max_length)
    encoder = Model(x, z)
    encoded_input = Input(shape=(latent_rep_dim,))
    decoder = Model(encoded_input, Decoder(encoded_input, latent_rep_size=latent_rep_dim,
                                           smile_max_length=smile_max_length,
                     charset_length=len(char_set)))
    x1 = Input(shape=(smile_max_length, len(char_set)), name='input_1')
    vae_loss, z1 = Encoder(x1, latent_rep_size=latent_rep_dim, smile_max_length=smile_max_length)
    autoencoder = Model(x1, Decoder(z1, latent_rep_size=latent_rep_dim,
                                           smile_max_length=smile_max_length,
                     charset_length=len(char_set)))
    autoencoder.compile(optimizer='Adam', loss=vae_loss, metrics =['accuracy'])
    return autoencoder


# 3 epoch training.
# 1. 1mil GDB17.
# 2. 500K with a 50/50 split between GDB and cations.
# 3. all bootstrapped cations.

# In[20]:


latent_dim_sizes = list(range(287, 298))
for dim in latent_dim_sizes:
    autoencoder = create_network(latent_rep_dim = dim)
    epoch1hist = autoencoder.fit(X_train1, X_train1, shuffle=False, validation_data=(X_test1, X_test1))
    epoch2hist = autoencoder.fit(X_train2, X_train2, shuffle=False, validation_data=(X_test2, X_test2))
    epoch3hist = autoencoder.fit(X_train3, X_train3, shuffle=False, validation_data=(X_test3, X_test3))
    autoencoder.save('latent_size_{}.h5'.format(dim))
    
    with open('latent_size_{}.json'.format(dim), 'w') as f:
        dump = {'epoch_histories': [{'epoch': 1, 'history': epoch1hist.history}, 
                                    {'epoch': 2, 'history': epoch2hist.history}, 
                                    {'epoch': 3, 'history': epoch3hist.history}
                                   ]
               }
        json.dump(dump, f)

