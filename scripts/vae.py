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
from keras.layers import Dense, Dropout, Input, Multiply, Add, Lambda, concatenate
from keras.layers.core import Dense, Activation, Flatten, RepeatVector
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import GRU
from keras.layers.convolutional import Convolution1D

##plot
#import matplotlib.pylab as plt
#import seaborn as sns; sns.set()
#from matplotlib import colors
#from itertools import cycle

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
               max_length = 62,
               latent_rep_size = 292,
               weights_file = None,
               qspr = False,
               mol_inputs=1,
               conv_layers=3,
               gru_layers=3,
               conv_epsilon_std=0.01,
               conv_filters=3, 
               conv_kernel_size=9,
               gru_output_units=501):
        
        charset_length = len(charset)
        
        if mol_inputs == 1:
            x = Input(shape=(max_length, charset_length), name='one_hot_input_encoder')
        elif mol_inputs ==2:
            x1 = Input(shape=(max_length, charset_length), name='one_hot_cation_input')
            x2 = Input(shape=(max_length, charset_length), name='one_hot_anion_input')
            x = [x1, x2]
        
        ###ENCODER/DECODER
        _, z = self._buildEncoder(x, 
                                  latent_rep_size, 
                                  max_length,
                                  conv_epsilon_std,
                                  conv_layers, 
                                  conv_filters, 
                                  conv_kernel_size,
                                  mol_inputs)
        self.encoder = Model(x, z)
        encoded_input = Input(shape=(latent_rep_size,), name='encoded_input')
        self.decoder = Model(
            encoded_input,
            self._buildDecoder(
                encoded_input,
                latent_rep_size,
                max_length,
                charset_length,
                gru_layers,
                gru_output_units,
                mol_inputs
            )
        )

        ###AUTOENCODER
        vae_loss, z1 = self._buildEncoder(x, 
                                          latent_rep_size, 
                                          max_length,
                                          conv_epsilon_std,
                                          conv_layers, 
                                          conv_filters,
                                          conv_kernel_size,
                                          mol_inputs)
        
        if qspr:
            self.autoencoder = Model(
                x,
                self._buildDecoderQSPR(
                    z1,
                    latent_rep_size,
                    max_length,
                    charset_length,
                    gru_layers,
                    gru_output_units,
                    mol_inputs
                )
            )

        else:
            self.autoencoder = Model(
                x,
                self._buildDecoder(
                    z1,
                    latent_rep_size,
                    max_length,
                    charset_length,
                    gru_layers,
                    gru_output_units,
                    mol_inputs
                )
            )
            
        self.qspr = Model(
            x,
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
            if mol_inputs == 1:
                self.autoencoder.compile(optimizer = 'Adam',
                                         loss = {'decoded_mean': vae_loss, 
                                                 'qspr': 'mean_squared_error'},
                                         metrics = ['accuracy', 'mse'])
            elif mol_inputs == 2:
                self.autoencoder.compile(optimizer = 'Adam',
                                         loss = {'decoded_cation_mean': vae_loss, 
                                                 'decoded_anion_mean': vae_loss, 
                                                 'qspr': 'mean_squared_error'},
                                         metrics = ['accuracy', 'mse'])
        else:
            if mol_inputs == 1:
                self.autoencoder.compile(optimizer = 'Adam',
                                         loss = {'decoded_mean': vae_loss},
                                         metrics = ['accuracy'])
            elif mol_inputs == 2:
                self.autoencoder.compile(optimizer = 'Adam',
                                         loss = {'decoded_cation_mean': vae_loss,
                                                 'decoded_anion_mean': vae_loss},
                                         metrics = ['accuracy'])
    def _buildEncoder(self, x, latent_rep_size, max_length, epsilon_std = 0.01, conv_layers=3, 
                      conv_filters=9, conv_kernel_size=9, mol_inputs=1):
        if mol_inputs == 2:
            x = concatenate(x)
        h = Convolution1D(9, 9, activation = 'relu', name='conv_1')(x)
        for convolution in range(conv_layers-2):
            h = Convolution1D(conv_filters, conv_kernel_size, activation = 'relu', name='conv_{}'.
                              format(convolution+2))(h)
        h = Convolution1D(10, 11, activation = 'relu', name='conv_{}'.format(conv_layers))(h)
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

    def _buildDecoderQSPR(self, z, latent_rep_size, max_length, charset_length, 
                          gru_layers=3, gru_output_units=501, mol_inputs=1):

        h = Dense(latent_rep_size, name='latent_input', activation = 'relu')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='gru_1')(h)
        for gru in range(gru_layers-2):
            h = GRU(gru_output_units, return_sequences = True, name='gru_{}'.format(gru+2))(h)
        h = GRU(501, return_sequences = True, name='gru_{}'.format(gru_layers))(h)
        if mol_inputs == 1:
            smiles_decoded = TimeDistributed(Dense(charset_length, activation='softmax'), 
                                             name='decoded_mean')(h)
        elif mol_inputs == 2:
            cation_decoded = TimeDistributed(Dense(charset_length, activation='softmax'), 
                                             name='decoded_cation_mean')(h)
            anion_decoded = TimeDistributed(Dense(charset_length, activation='softmax'), 
                                            name='decoded_anion_mean')(h)

        h = Dense(latent_rep_size, name='qspr_input', activation='relu')(z)
        h = Dense(100, activation='relu', name='hl_1')(h)
        h = Dropout(0.5)(h)
        smiles_qspr = Dense(1, activation='linear', name='qspr')(h)
        
        if mol_inputs == 1:
            return smiles_decoded, smiles_qspr
        elif mol_inputs == 2:
            return cation_decoded, anion_decoded, smiles_qspr

    def _buildDecoder(self, z, latent_rep_size, max_length, charset_length, 
                      gru_layers=3, gru_output_units=501, mol_inputs=1):

        h = Dense(latent_rep_size, name='latent_input', activation = 'relu')(z)
        h = RepeatVector(max_length, name='repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='gru_1')(h)
        for gru in range(gru_layers-2):
            h = GRU(gru_output_units, return_sequences = True, name='gru_{}'.format(gru+2))(h)
        h = GRU(501, return_sequences = True, name='gru_{}'.format(gru_layers))(h)
        if mol_inputs == 1:
            smiles_decoded = TimeDistributed(Dense(charset_length, activation='softmax'), 
                                             name='decoded_mean')(h)
        elif mol_inputs ==2:
            cation_decoded = TimeDistributed(Dense(charset_length, activation='softmax'), 
                                             name='decoded_cation_mean')(h)
            anion_decoded = TimeDistributed(Dense(charset_length, activation='softmax'), 
                                            name='decoded_anion_mean')(h)

        if mol_inputs == 1:
            return smiles_decoded
        elif mol_inputs == 2:
            return cation_decoded, anion_decoded
    
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
