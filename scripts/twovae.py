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

class TwoMoleculeVAE():

    cation_autoencoder = None
    anion_autoencoder = None
    
    def create(self,
               cat_charset,
               ani_charset,
               cat_max_length = 51,
               cat_weights_file = None,
               ani_max_length = 62,
               ani_weights_file = None,
               latent_rep_size = 292,
               qspr_weights_file = None,
               qspr = False):
        
        ### cation
        max_length = cat_max_length
        charset = cat_charset
        charset_length = len(charset)
        x = Input(shape=(max_length, charset_length))
        _, z = self._buildEncoder('cation',
                                  x, 
                                  latent_rep_size,
                                  max_length)
        self.cation_encoder = Model(x, z)
            
        encoded_input = Input(shape=(latent_rep_size,))
        self.cation_decoder = Model(
            encoded_input,
            self._buildDecoder('cation',
                               encoded_input,
                               latent_rep_size,
                               max_length,
                               charset_length
            )
        )

        ### anion
        max_length = ani_max_length
        charset = ani_charset
        charset_length = len(ani_charset)
        x = Input(shape=(max_length, charset_length))
        _, z = self._buildEncoder('anion',
                                  x, 
                                  latent_rep_size,
                                  max_length)
        self.cation_encoder = Model(x, z)
            
        encoded_input = Input(shape=(latent_rep_size,))
        self.cation_decoder = Model(
            encoded_input,
            self._buildDecoder('anion',
                               encoded_input,
                               latent_rep_size,
                               max_length,
                               charset_length
            )
        )
        ###AUTOENCODER
        max_length = cat_max_length
        charset = cat_charset
        charset_length = len(charset)
        cat_x1 = Input(shape=(max_length, charset_length))
        cat_vae_loss, cat_z1 = self._buildEncoder('cation', cat_x1, latent_rep_size, max_length)
        
        max_length = ani_max_length
        charset = ani_charset
        charset_length = len(ani_charset)
        ani_x1 = Input(shape=(max_length, charset_length))
        ani_vae_loss, ani_z1 = self._buildEncoder('anion', ani_x1, latent_rep_size, max_length)
        
        
        if qspr:
            self.autoencoder = Model(
                [cat_x1, ani_x1],
                self._buildAutoencoderQSPR(
                    cat_z1,
                    ani_z1,
                    latent_rep_size,
                    cat_max_length,
                    len(cat_charset),
                    ani_max_length,
                    len(ani_charset)
                )
            )
        else:
            self.autoencoder = Model(
                [cat_x1, ani_x1],
                self._buildAutoencoder(
                    cat_z1,
                    ani_z1,
                    latent_rep_size,
                    cat_max_length,
                    len(cat_charset),
                    ani_max_length,
                    len(ani_charset)
                )
            )
        
        ###QSPR
        self.qspr = Model(
            [cat_x1, ani_x1],
            self._buildQSPR(
                cat_z1,
                ani_z1,
                latent_rep_size,
                max_length,
                charset_length
            )
        )
        

        if cat_weights_file:
            self.cation_encoder.load_weights(cat_weights_file, by_name = True)
            self.cation_decoder.load_weights(cat_weights_file, by_name = True)
        if ani_weights_file:
            self.anion_encoder.load_weights(ani_weights_file, by_name = True)
            self.anion_decoder.load_weights(ani_weights_file, by_name = True)
#         if qspr_weights_file:
#             self.qspr.load_weights(weights_file, by_name = True)
        if qspr:
            self.autoencoder.compile(optimizer = 'Adam',
                                     loss = {'cation_decoded_mean': cat_vae_loss,
                                             'anion_decoded_mean': ani_vae_loss,
                                             'qspr': 'mean_squared_error'},
                                     metrics = ['accuracy', 'mse'])
        else:
            self.autoencoder.compile(optimizer = 'Adam',
                                     loss = {'cation_decoded_mean': cat_vae_loss,
                                             'anion_decoded_mean': ani_vae_loss},
                                     metrics = ['accuracy'])
            
            
    def _buildEncoder(self, my_name, x, latent_rep_size, max_length, epsilon_std = 0.01):
        h = Convolution1D(9, 9, activation = 'relu', name='{}_conv_1'.format(my_name))(x)
        h = Convolution1D(9, 9, activation = 'relu', name='{}_conv_2'.format(my_name))(h)
        h = Convolution1D(10, 11, activation = 'relu', name='{}_conv_3'.format(my_name))(h)
        h = Flatten(name='{}_flatten_1'.format(my_name))(h)
        h = Dense(435, activation = 'relu', name='{}_dense_1'.format(my_name))(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), 
                                      mean=0., stddev = epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='{}_z_mean'.format(my_name), 
                       activation = 'linear')(h)
        z_log_var = Dense(latent_rep_size, name='{}_z_log_var'.format(my_name), 
                          activation = 'linear')(h)

        def vae_loss(x, x_decoded_mean):
            x = K.flatten(x)
            x_decoded_mean = K.flatten(x_decoded_mean)
            xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
            return xent_loss + kl_loss

        return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), 
                                 name='{}_lambda'.format(my_name))([z_mean, z_log_var]))

    def _buildAutoencoderQSPR(self, cat_z, ani_z, latent_rep_size, 
                          cat_max_length, cat_charset_length,
                          ani_max_length, ani_charset_length):

        h = Dense(latent_rep_size, name='cation_latent_input', activation = 'relu')(cat_z)
        h = RepeatVector(cat_max_length, name='cation_repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='cation_gru_1')(h)
        h = GRU(501, return_sequences = True, name='cation_gru_2')(h)
        h = GRU(501, return_sequences = True, name='cation_gru_3')(h)
        cat_smiles_decoded = TimeDistributed(Dense(cat_charset_length, 
                                                   activation='softmax'), 
                                             name='cation_decoded_mean')(h)
        
        h = Dense(latent_rep_size, name='anion_latent_input', activation = 'relu')(ani_z)
        h = RepeatVector(ani_max_length, name='anion_repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='anion_gru_1')(h)
        h = GRU(501, return_sequences = True, name='anion_gru_2')(h)
        h = GRU(501, return_sequences = True, name='anion_gru_3')(h)
        ani_smiles_decoded = TimeDistributed(Dense(ani_charset_length, 
                                                   activation='softmax'), 
                                             name='anion_decoded_mean')(h)
        
        combined = concatenate([cat_z, ani_z])
        h = Dense(latent_rep_size*2, name='qspr_input', activation='relu')(combined)
        h = Dense(100, activation='relu', name='hl_1')(h)
        h = Dropout(0.5)(h)
        smiles_qspr = Dense(1, activation='linear', name='qspr')(h)

        return cat_smiles_decoded, ani_smiles_decoded, smiles_qspr

    def _buildAutoencoder(self, cat_z, ani_z, latent_rep_size, 
                          cat_max_length, cat_charset_length,
                          ani_max_length, ani_charset_length):

        h = Dense(latent_rep_size, name='cation_latent_input', activation = 'relu')(cat_z)
        h = RepeatVector(cat_max_length, name='cation_repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='cation_gru_1')(h)
        h = GRU(501, return_sequences = True, name='cation_gru_2')(h)
        h = GRU(501, return_sequences = True, name='cation_gru_3')(h)
        cat_smiles_decoded = TimeDistributed(Dense(cat_charset_length, 
                                                   activation='softmax'), 
                                             name='cation_decoded_mean')(h)
        
        h = Dense(latent_rep_size, name='anion_latent_input', activation = 'relu')(ani_z)
        h = RepeatVector(ani_max_length, name='anion_repeat_vector')(h)
        h = GRU(501, return_sequences = True, name='anion_gru_1')(h)
        h = GRU(501, return_sequences = True, name='anion_gru_2')(h)
        h = GRU(501, return_sequences = True, name='anion_gru_3')(h)
        ani_smiles_decoded = TimeDistributed(Dense(ani_charset_length, 
                                                   activation='softmax'), 
                                             name='anion_decoded_mean')(h)

        return cat_smiles_decoded, ani_smiles_decoded
    
    def _buildDecoder(self, my_name, z, latent_rep_size, max_length, charset_length):

        h = Dense(latent_rep_size, name='{}_latent_input'.format(my_name), activation = 'relu')(z)
        h = RepeatVector(max_length, name='{}_repeat_vector'.format(my_name))(h)
        h = GRU(501, return_sequences = True, name='{}_gru_1'.format(my_name))(h)
        h = GRU(501, return_sequences = True, name='{}_gru_2'.format(my_name))(h)
        h = GRU(501, return_sequences = True, name='{}_gru_3'.format(my_name))(h)
        smiles_decoded = TimeDistributed(Dense(charset_length, activation='softmax'),
                                         name='{}_decoded_mean'.format(my_name))(h)

        return smiles_decoded
    
    def _buildQSPR(self, cat_z, ani_z, latent_rep_size, max_length, charset_length):
        combined = concatenate([cat_z, ani_z])
        h = Dense(latent_rep_size*2, name='qspr_input', activation='relu')(combined)
        h = Dense(100, activation='relu', name='hl_1')(h)
        h = Dropout(0.5)(h)
        return Dense(1, activation='linear', name='qspr')(h)

    def save(self, filename):
        self.autoencoder.save_weights(filename)
    
    def load(self, charset, weights_file, latent_rep_size = 292):
        self.create(charset, weights_file = weights_file, latent_rep_size = latent_rep_size)