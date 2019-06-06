##plot
#import matplotlib.pylab as plt
#import seaborn as sns; sns.set()
#from matplotlib import colors
#from itertools import cycle

#data
import pandas as pd
import numpy as np
from numpy.linalg import norm
import json
import random
import copy

#ML
import keras
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
from keras import objectives
from keras.objectives import binary_crossentropy #objs or losses
from keras.layers import Dense, Dropout, Input, Multiply, Add, Lambda
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D

#chem
#import salty
#from rdkit import Chem
#from rdkit.Chem.Fingerprints import FingerprintMols
#from rdkit import DataStructs
#from rdkit.Chem import Draw


class MoleculeVAE():

    autoencoder = None
    
    def create(self,
               charset,
               max_length = 105,
               latent_rep_size = 292,
               weights_file = None,
               qspr = False):
        charset_length = len(charset)
    
        x = Input(shape=(max_length, charset_length))
        _, z = self._buildEncoder(x, latent_rep_size, max_length)
        self.encoder = Model(x, z)
        encoded_input = Input(shape=(latent_rep_size,))
        self.decoder = Model(
            encoded_input,
            self._buildDecoder(
                encoded_input,
                latent_rep_size,
                max_length,
                charset_length
            )
        )

        x1 = Input(shape=(max_length, charset_length))
        vae_loss, z1 = self._buildEncoder(x1, latent_rep_size, max_length)
        
        if qspr:
            self.autoencoder = Model(
                x1,
                self._buildDecoderQSPR(
                    z1,
                    latent_rep_size,
                    max_length,
                    charset_length
                )
            )

        else:
            self.autoencoder = Model(
                x1,
                self._buildDecoder(
                    z1,
                    latent_rep_size,
                    max_length,
                    charset_length
                )
            )
            
        self.qspr = Model(
            x1,
            self._buildQSPR(
                z1,
                latent_rep_size,
                max_length,
                charset_length
            )
        )


        if weights_file:
            self.autoencoder.load_weights(weights_file, by_name = True)
            self.encoder.load_weights(weights_file, by_name = True)
            self.decoder.load_weights(weights_file, by_name = True)
            self.qspr.load_weights(weights_file, by_name = True)
        if qspr:
            self.autoencoder.compile(optimizer = 'Adam',
                                     loss = {'decoded_mean': vae_loss, 'qspr': 'mean_squared_error'},
                                     metrics = ['accuracy', 'mse'])
        else:
            self.autoencoder.compile(optimizer = 'Adam',
                                     loss = {'decoded_mean': vae_loss},
                                     metrics = ['accuracy'])
    def _buildEncoder(self, x, latent_rep_size, max_length, epsilon_std = 0.01):
        h = Convolution1D(9, 9, activation = 'relu', name='conv_1')(x)
        h = Convolution1D(9, 9, activation = 'relu', name='conv_2')(h)
        h = Convolution1D(10, 11, activation = 'relu', name='conv_3')(h)
        h = Flatten(name='flatten_1')(h)
        h = Dense(435, activation = 'relu', name='dense_1')(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev = epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))

    def _buildDecoderQSPR(self, z, latent_rep_size, max_length, charset_length):

        h = Dense(latent_rep_size, name='latent_input', activation = 'relu')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='gru_1')(h)
        h = GRU(501, return_sequences = True, name='gru_2')(h)
        h = GRU(501, return_sequences = True, name='gru_3')(h)
        smiles_decoded = TimeDistributed(Dense(charset_length, activation='softmax'), name='decoded_mean')(h)

        h = Dense(latent_rep_size, name='qspr_input', activation='relu')(z)
        h = Dense(100, activation='relu', name='hl_1')(h)
        h = Dropout(0.5)(h)
        smiles_qspr = Dense(1, activation='linear', name='qspr')(h)

        return smiles_decoded, smiles_qspr

    def _buildDecoder(self, z, latent_rep_size, max_length, charset_length):

        h = Dense(latent_rep_size, name='latent_input', activation = 'relu')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='gru_1')(h)
        h = GRU(501, return_sequences = True, name='gru_2')(h)
        h = GRU(501, return_sequences = True, name='gru_3')(h)
        smiles_decoded = TimeDistributed(Dense(charset_length, activation='softmax'), name='decoded_mean')(h)

        return smiles_decoded
    
    def _buildQSPR(self, z, latent_rep_size, max_length, charset_length):
        h = Dense(latent_rep_size, name='latent_input', activation='relu')(z)
        h = Dense(100, activation='relu', name='hl_1')(h)
        h = Dropout(0.5)(h)
        return Dense(1, activation='linear', name='qspr')(h)

    def save(self, filename):
        self.autoencoder.save_weights(filename)
    
    def load(self, charset, weights_file, latent_rep_size = 292):
        self.create(charset, weights_file = weights_file, latent_rep_size = latent_rep_size)


def my_colors():
    """
    return a tableau colors iterable
    """
    tab = cycle(colors.TABLEAU_COLORS)
    return tab


def pad_smiles(smiles_string, smile_max_length):
    """
    pad smiles string with whitespace up to
    smile_max_length
    """
    if len(smiles_string) < smile_max_length:
        return smiles_string + " " * (smile_max_length - len(smiles_string))


def one_hot(smi, char_to_index, smile_max_length=105):
    """
    one not encode input smiles according to char_to_index
    and smile_max_length
    """
    test_smi = smi
    char_set = set(char_to_index.keys())
    test_smi = pad_smiles(test_smi, smile_max_length)
    Z = np.zeros((1, smile_max_length, len(list(char_set))), dtype=np.bool)
    for t, char in enumerate(test_smi):
        Z[0, t, char_to_index[char]] = 1
    return Z


def sample(a, temperature=1.0):
    """
    helper function to sample an index from a probability array
    work around from https://github.com/llSourcell/How-to-Generate-Music-Demo/issues/4
    """
    a = np.log(a) / temperature 
    dist = np.exp(a)/np.sum(np.exp(a)) 
    choices = range(len(a)) 
    return np.random.choice(choices, p=dist)


def decode_smiles(vae, smi, char_to_index, temp=0.5, smile_max_length=105):
    """
    vae: variational autoencoder to encode/decode input
    smi: smiles string to encode
    temp: temperature at which to perform sampling
    """
    char_list = list(char_to_index.keys())
    index_to_char = dict((i, c) for i, c in enumerate(char_list))
    smi = pad_smiles(smi, smile_max_length)
    autoencoder = vae
    Z = np.zeros((1, smile_max_length, len(char_list)), dtype=np.bool)
    for t, char in enumerate(smi):
        Z[0, t, char_to_index[char]] = 1
    string = ""
    for i in autoencoder.predict(Z):
        for j in i:
            index = sample(j, temperature=temp)
            string += index_to_char[index]
    return string


def decode_latent(vae, z, char_to_index, temp=0.5, smile_max_length=51):
    """
    vae: variational autoencoder to encode/decode input
    z: encoded smiles str
    temp: temperature at which to perform sampling
    """
    char_list = list(char_to_index.keys())
    index_to_char = dict((i, c) for i, c in enumerate(char_list))
    string = ""
    for i in vae.decoder.predict(z):
        for j in i:
            index = sample(j, temperature=temp)
            string += index_to_char[index]
    return string


def slerp(p0, p1, t):
    """
    return spherical linear interpolation coordinates between
    points p0 and p1 at t intervals
    """
    omega = np.arccos(np.dot(p0/norm(p0), p1/norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1
    

def interpolate_structures(vae, ps, char_to_index, limit=1e4, write=False, temp=0.5):
    """
    Quick and Dirty: 
    Use this VAE, these interpolations of embeded z's, and this char_to_index
    dictionary to randomly generate structures at temp
    """
    rdkit_mols = []
    temps = []
    iterations = []
    iteration = limit_counter = 0
    df = pd.DataFrame()
    for p in ps:
        while True:
            iteration += 1
            limit_counter += 1
            t = temp
            candidate = decode_latent(chemvae, p.reshape(1,292), char_to_index, temp=t)
            try:
                sampled = Chem.MolFromSmiles(candidate)
                cation = Chem.AddHs(sampled)
                Chem.EmbedMolecule(cation, Chem.ETKDG())
                Chem.UFFOptimizeMolecule(cation)
                cation = Chem.RemoveHs(cation)
                candidate = Chem.MolToSmiles(cation)
                if candidate not in rdkit_mols:
                    temps.append(t)
                    iterations.append(iteration)
                    rdkit_mols.append(candidate) 
                    limit_counter = 0
                    df = pd.DataFrame([rdkit_mols,temps,iterations]).T
                    df.columns = ['smiles', 'temperature', 'iteration']
                    print(df)
                    print(t)
                    break
            except:
                pass
            if limit_counter > limit:
                break
        if write:
            df = pd.DataFrame([rdkit_mols,temps,iterations]).T
            df.columns = ['smiles', 'temperature', 'iteration']
            pd.DataFrame.to_csv(df, path_or_buf='{}.csv'.format(write), index=False)
    return df


def generate_structures(vae, smi, char_to_index, limit=1e4, write=False):
    """
    Quick and Dirty: 
    Use this VAE, this smiles string, and this char_to_index
    dictionary to randomly generate structures at random temperatures
    """
    rdkit_mols = []
    temps = []
    iterations = []
    iteration = limit_counter = 0
    while True:
        iteration += 1
        limit_counter += 1
        t = random.random()*2
        candidate = decode_smiles(vae, smi, char_to_index, temp=t).split(" ")[0]
        try:
            sampled = Chem.MolFromSmiles(candidate)
            cation = Chem.AddHs(sampled)
            Chem.EmbedMolecule(cation, Chem.ETKDG())
            Chem.UFFOptimizeMolecule(cation)
            cation = Chem.RemoveHs(cation)
            candidate = Chem.MolToSmiles(cation)
            if candidate not in rdkit_mols:
                temps.append(t)
                iterations.append(iteration)
                rdkit_mols.append(candidate) 
                limit_counter = 0
                df = pd.DataFrame([rdkit_mols,temps,iterations]).T
                df.columns = ['smiles', 'temperature', 'iteration']
                print(df)
        except:
            pass
        if limit_counter > limit:
            break
        if write:
            df = pd.DataFrame([rdkit_mols,temps,iterations]).T
            df.columns = ['smiles', 'temperature', 'iteration']
            pd.DataFrame.to_csv(df, path_or_buf='{}.csv'.format(write), index=False)
    return df
