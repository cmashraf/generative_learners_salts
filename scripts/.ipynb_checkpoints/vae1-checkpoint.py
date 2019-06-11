import matplotlib.pylab as plt
import numpy as np
import seaborn as sns; sns.set()
import json

import salty
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from random import shuffle
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

from rdkit import Chem
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs
from rdkit.Chem import Draw

import os
import sys
sys.path.insert(0, '../')
from scripts import build_vae, decode_smiles

class suppress_rdkit_sanity(object):
    """
    Context manager for doing a "deep suppression" of stdout and stderr
    during certain calls to RDKit.
    """
    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
            

vae_models = ['Bootstrap_250k_Cation_1.h5', 'Bootstrap_1Mil_Cation_1.h5',
    '1Mil_GDB17.h5', '1Mil_GDB17_split_500k.h5', '1Mil_GDB17_split_500k_cation_500k.h5']

f = open("../data/1mil_GDB17.json","r")
char_to_index = json.loads(f.read())
autoencoder = build_vae()
smi = 'CCCC[n+]1ccc(cc1)C'
autoencoder.load_weights("../data/{}".format(vae_models[0]))

def generate_structures(vae, smi, char_to_index, limit=1e4, write=False):
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

# with suppress_rdkit_sanity():
generate_structures(autoencoder, smi, char_to_index, limit=1e4, write='vae_1_structures')
