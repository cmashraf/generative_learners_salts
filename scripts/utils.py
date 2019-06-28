#misc
import os
from os.path import join
from itertools import combinations 
from keras.models import load_model
import keras

#data
import pandas as pd
import numpy as np
from numpy.linalg import norm
import json
import random
import copy

#plot
from IPython.display import clear_output, display
import matplotlib.pylab as plt
import seaborn as sns; sns.set()
from matplotlib import colors
from itertools import cycle

#chem
import salty
import gains as genetic
from rdkit import Chem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs
from rdkit.Chem import Draw
from rdkit.ML.Descriptors.MoleculeDescriptors import\
    MolecularDescriptorCalculator as calculator

class VAESaver(keras.callbacks.Callback):
    """
    custom callback to save vae models at 10 and 30 epochs
    """
    def on_epoch_end(self, epoch, logs={}):
        if epoch == 10:  # or save after some epoch, each k-th epoch etc.
            self.model.save("{}_{}.h5".format(self.model.name, epoch))
        elif epoch == 30:  
            self.model.save("{}_{}.h5".format(self.model.name, epoch))
        elif epoch == 100:
            self.model.save("{}_{}.h5".format(self.model.name, epoch))

            
def rSubset(arr, r): 
  
    # return list of all subsets of length r 
    # to deal with duplicate subsets use  
    # set(list(combinations(arr, r))) 
    return list(combinations(arr, r)) 


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


def decode_latent(decoder, z, char_to_index, temp=0.5, smile_max_length=51):
    """
    vae: variational autoencoder to encode/decode input
    z: encoded smiles str
    temp: temperature at which to perform sampling
    """
    char_list = list(char_to_index.keys())
    index_to_char = dict((i, c) for i, c in enumerate(char_list))
    string = ""
    for i in decoder.predict(z):
        for j in i:
            index = sample(j, temperature=temp)
            string += index_to_char[index]
    return string

def interpolate_structures(decoder, ps, char_to_index, limit=1e4, write=False, temp=0.5,
                          verbose=0):
    """
    Quick and Dirty: 
    Use this decoder, these interpolations of embeded z's, and this char_to_index
    dictionary to randomly generate structures at temp
    """
    rdkit_mols = []
    temps = []
    iterations = []
    df = pd.DataFrame()
    total_iterations = 0
    for p in ps:
        iteration = limit_counter = 0
        while True:
            total_iterations += 1
            iteration += 1
            limit_counter += 1
            t = temp
            candidate = decode_latent(decoder, p.reshape(1,292), char_to_index, temp=t)
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
                    if verbose == 0:
                        clear_output(wait=True)
                        print(df)
                        print('total iterations:\t {}'.format(total_iterations))
                    elif verbose == 1:
                        clear_output(wait=True)
                        print('total iterations:\t {}'.format(total_iterations))
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


def slerp(p0, p1, t):
    """
    return spherical linear interpolation coordinates between
    points p0 and p1 at t intervals
    """
    omega = np.arccos(np.dot(p0/norm(p0), p1/norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0-t)*omega) / so * p0 + np.sin(t*omega)/so * p1
    

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
            
def get_fitness(anion, genes, target, models, deslists):
    """
    the fitness function passed to the engine. models refer
    to rdkit/keras models.

    Parameters
    ----------
    anion : RDKit Mol Object
        the anion comprising the IL
    genes : str
        the smiles string representing the cation of the IL
    target : list, int, or array
        the target property values of the IL
    models : array of, or single Keras model
        array or single keras model to use in the prediction
        of the targets
    deslists : array of, or single pandas dataFrame
        contains the mean and stds of the model inputs
        
    Returns
    -------
    score: float
        percent deviation from target
    predictions: array of floats
        array of hard predictions from
        qspr models
    """
    predictions = []
    for i, name in enumerate(models):
        cation = Chem.MolFromSmiles(genes)
        model = name
        deslist = deslists[i]
        if isinstance(deslist, list):
            deslist = deslist[0]
        feature_vector = []

        for item in deslist:

            if "anion" in item:
                with suppress_rdkit_sanity():
                    feature_vector.append(calculator([item.partition('-')
                                          [0]]).CalcDescriptors(anion)[0])
            elif "cation" in item:
                with suppress_rdkit_sanity():
                    feature_vector.append(calculator([item.partition('-')
                                          [0]]).CalcDescriptors(cation)[0])
            elif "Temperature, K" in item:
                feature_vector.append(298.15)
            elif "Pressure, kPa" in item:
                feature_vector.append(101.325)
            else:
                print("unknown descriptor in list: %s" % item)
        features_normalized = (feature_vector - deslist.iloc[0].values) /\
            deslist.iloc[1].values
        prediction = np.round(np.exp(model.predict(np.array(
                              features_normalized).reshape(1, -1))[0]),
                              decimals=5)
        predictions.append(prediction[0])
    predictions = np.array(predictions)
    error = abs((predictions - target) / target)
    error = np.average(error)

    return 1 - error, predictions

def return_top_cations(prop, n=10, return_min_values=True, T = [297, 316], P = [99, 102]):
    """
    Returns salt and cation list containin n unique cations
    
    Parameters
    ----------
    prop: list
        list of properties to grab salts from
    n: int
        number of unique cations desired
    return_min_values: def True
        returns the minimum value of the (descending) sorted
        salts. if False, salts are sorted in ascending order
        and the max value is returned.
        
    Returns
    -------
    salts: list
        list of salts screened by property values
    cations: list
        list of cations in salts list
    values: float
        min/max property value depending on return_min_values
    """   
    devmodel = salty.aggregate_data(prop, T=T, P=P, merge='Union')
    devmodel.Data['smiles_string'] = devmodel.Data['smiles-cation'] + "." + devmodel.Data['smiles-anion']
    salts = devmodel.Data['smiles_string']
    salts = salts.reset_index(drop=True)

    property_index = 6
    if prop == 'melting_point':
        property_index = 8
    print(devmodel.Data.columns[-property_index])
    print("total salts in training data:\t{}".format(salts.shape[0]))
    df = devmodel.Data
    salts = df.sort_values(devmodel.Data.columns[-property_index], ascending=False)['smiles_string']
    while True:
        for i in range(len(salts)):
            if return_min_values == True:
                salts = df.sort_values(devmodel.Data.columns[-property_index], 
                                       ascending=False)['smiles_string']\
                                       [:i].unique()
                values = np.exp(df.sort_values(devmodel.Data.columns[-property_index], 
                                            ascending=False)[devmodel.Data.columns\
                                                [-property_index]][:i].unique())
            else:
                salts = df.sort_values(devmodel.Data.columns[-property_index], 
                                       ascending=True)['smiles_string']\
                                       [:i].unique()
                values = np.exp(df.sort_values(devmodel.Data.columns[-property_index], 
                                            ascending=True)[devmodel.Data.columns\
                                                [-property_index]][:i].unique())
            cations = pd.DataFrame([j.split('.')[0] for j in salts], 
                                   columns=['salt'])['salt'].unique()
            if len(cations) == n:
                print('unique salts:\t{}'.format(len(salts)))
                print('unique cations:\t{}'.format(len(cations)))
                print('min/max values:\t{:.4f}, {:.4f}'.format(min(values), max(values)))
                print('')
                if return_min_values == True:
                    print('salts sorted in descending order and the minimum value of the top 10 unique cations was returned')
                    return salts, cations, min(values)
                else:
                    print('salts sorted in ascending order and the maximum value of the top 10 unique cations was returned')
                    return salts, cations, max(values)
                
from rdkit.Chem import AllChem as Chem

def generate_solvent_vae(vae, char_to_index, smile_max_length, salts, model_ID=None, target=None, qspr=False, find=100, optimalCutOff=None,
                         greaterThanCutOff=True, md_model=None, path=None, desired_fitness=0.01, verbose=1, sanitize_cut_off=1e4):
    """
    returns dictionary of solvents with targeted properties
    
    Parameters
    ----------
    vae: keras model
        vae generator
    salts: list
        list of salts to grab cation and anion seeds
    model_ID: list, default None
        salty (rdkit qspr) model name to envoke for prop prediction
        and fitness if desired
    target: float or int, default None
        property target to interact with fitness function
    qspr: boolean, default False
        if True will include vae qspr prediction in returned
        di
    find: int, default 100
        number of desired hits
    optimalCutOff: float or int
        max/min of desired property
    greaterThanCutOff: boolean. default True
        if True will return hits higher than optimalCutOff
        if False will return hits lower than optimalCutOff
    md_model:
        if True will return md supplemented rdkit qspr
        predictions
    path:
        relative location of the md rdkit qspr files
    desired_fitness: float, default 0.01
        if not optimalCutOff then will return salt w/in
        error of desired_fitness
    verbose: default 1,
        desired verbosity 
        
    Returns
    -------
    found_di: pandas dataframe
        dictionary of found salts
    """
    
    ### setup
    char_list = list(char_to_index.keys())
    index_to_char = dict((i, c) for i, c in enumerate(char_list))
    found_di = {'salt': [],
                    'cat seed': [],
                 'ani seed': [],
                 'temperature': [],
                  'candidate': [],
                  'attempts': []}
    if model_ID is not None:
        found_di.update(
                {'rdkit qspr': []})
    if md_model is not None:
        found_di.update(
                 {'rdkit-md qspr': []})
    if qspr:
        found_di.update(
                 {'vae qspr': []})
    if model_ID is not None:
        for i, name in enumerate(model_ID):
            model = np.array([genetic.load_data("{}_qspr.h5".format(name),
                                                h5File=True)])
            deslist = list([genetic.load_data("{}_desc.csv".format(name))])
            summary = genetic.load_data("{}_summ.csv".format(name))

    path = '../data/'
    if md_model is not None:
        for i, name in enumerate(md_model):
            if path:
                model_md = np.array([load_model(join(path,
                                                  '{}_qspr.h5'.format(name)))])
                with open(join(path, '{}_desc.csv'.format(name)),
                          'rb') as csv_file:
                    deslist_md = list([pd.read_csv(csv_file, encoding='latin1')])
                with open(join(path, '{}_summ.csv'.format(name)),
                          'rb') as csv_file:
                    summary_md = pd.read_csv(csv_file, encoding='latin1')
    attempts = 0
    found = 0
    sanitize_attempts = 0
    ### begin search
    while True:
        current_found = found
        attempts += 1
        
        try:
            
            seed1 = salts[random.randint(0,len(salts)-1)].split('.')[0]
            seed2 = salts[random.randint(0,len(salts)-1)].split('.')[1]
            anion = Chem.MolFromSmiles(seed2)

            for rindex, i in enumerate(vae.autoencoder.predict([one_hot(seed1, char_to_index, smile_max_length=62),
                                                         one_hot(seed2, char_to_index, smile_max_length=62)])):
                string = ""
                if len(i.shape) > 2:
                    i = i[0] #for qspr chemvae there is an extra dim
                if rindex == 0:
                    temp=max(0.1,random.random()*2)
                    for j in i:
                        index = sample(j, temperature=temp)
                        string += index_to_char[index]    
                    sampled = Chem.MolFromSmiles(string)
                    cation = Chem.AddHs(sampled)
                    Chem.EmbedMolecule(cation, Chem.ETKDG())
                    Chem.UFFOptimizeMolecule(cation)
                    cation = Chem.RemoveHs(cation)
                    candidate = Chem.MolToSmiles(cation)
            molseed = Chem.MolFromSmiles(seed1)
            Chem.EmbedMolecule(molseed, Chem.ETKDG())
            Chem.UFFOptimizeMolecule(molseed)
            molseed = Chem.RemoveHs(molseed)
            molseedsmi = Chem.MolToSmiles(molseed)
        except:
            sanitize_attempts += 1
            continue 
        sanitize_attempts = 0
        if model_ID is not None and target is not None:
            scr, pre = get_fitness(anion, candidate, target, model,
                                        deslist)
            pre = pre[0]
        elif model_ID is not None: #we send a dummy variable to the fitness fn
            scr, pre = get_fitness(anion, candidate, 10, model,
                                        deslist)
            pre = pre[0]
        if md_model is not None and target is not None:
            scr_md, pre_md = get_fitness(anion, candidate, target, model_md,
                                        deslist_md)
            pre_md = pre_md[0]
        if molseedsmi != candidate:
            if candidate+'.'+seed2 not in found_di['salt']:
                if target is not None:
                    if optimalCutOff:
                        if greaterThanCutOff:
                            if pre >= target: #search with greater than target
                                if qspr:
                                    if verbose == 0:
                                        print("vae qspr output:\t{}".format(np.exp(i[0][0])))
                                    found_di['vae qspr'].append(np.exp(i[0][0]))
                                found_di['rdkit qspr'].append(pre)
                                found_di['salt'].append(candidate+'.'+seed2)
                                found_di['cat seed'].append(seed1)
                                found_di['ani seed'].append(seed2)
                                found_di['candidate'].append(candidate)
                                found_di['attempts'].append(attempts)
                                found_di['temperature'].append(temp)
                                if verbose == 0:
                                    print("rdkit qspr output:\t{}".format(pre))
                                    print("cat seed:\t{}".format(seed1))
                                    print("ani seed:\t{}".format(seed2))
                                    print("candidate:\t{}".format(candidate))
                                    print("attempts:\t{}".format(attempts))
                                if md_model is not None:
                                    if verbose == 0:
                                        print("rdkit-md qspr output:\t{}".format(pre_md))
                                    found_di['rdkit-md qspr'].append(pre_md)
                                found += 1
                        else:
                            if pre <= target: #search with less than target
                                if qspr:
                                    if verbose == 0:
                                        print("vae qspr output:\t{}".format(np.exp(i[0][0])))
                                    found_di['vae qspr'].append(np.exp(i[0][0]))
                                found_di['rdkit qspr'].append(pre)
                                found_di['salt'].append(candidate+'.'+seed2)
                                found_di['cat seed'].append(seed1)
                                found_di['ani seed'].append(seed2)
                                found_di['candidate'].append(candidate)
                                found_di['attempts'].append(attempts)
                                found_di['temperature'].append(temp)
                                if verbose == 0:
                                    print("rdkit qspr output:\t{}".format(pre))
                                    print("cat seed:\t{}".format(seed1))
                                    print("ani seed:\t{}".format(seed2))
                                    print("candidate:\t{}".format(candidate))
                                    print("attempts:\t{}".format(attempts))
                                if md_model is not None:
                                    if verbose == 0:
                                        print("rdkit-md qspr output:\t{}".format(pre_md))
                                    found_di['rdkit-md qspr'].append(pre_md)
                                found += 1
                    elif scr < desired_fitness: #search with specific target
                        if qspr:
                            if verbose == 0:
                                print("vae qspr output:\t{}".format(np.exp(i[0][0])))
                            found_di['vae qspr'].append(np.exp(i[0][0]))
                        found_di['rdkit qspr'].append(pre)
                        found_di['salt'].append(candidate+'.'+seed2)
                        found_di['cat seed'].append(seed1)
                        found_di['ani seed'].append(seed2)
                        found_di['candidate'].append(candidate)
                        found_di['attempts'].append(attempts)
                        found_di['temperature'].append(temp)
                        if verbose == 0:
                            print("rdkit qspr output:\t{}".format(pre))
                            print("cat seed:\t{}".format(seed1))
                            print("ani seed:\t{}".format(seed2))
                            print("candidate:\t{}".format(candidate))
                            print("attempts:\t{}".format(attempts))
                        if md_model is not None:
                            if verbose == 0:
                                print("rdkit-md qspr output:\t{}".format(pre_md))
                            found_di['rdkit-md qspr'].append(pre_md)
                        found += 1
                else: #search without target
                    found_di['salt'].append(candidate+'.'+seed2)
                    found_di['cat seed'].append(seed1)
                    found_di['ani seed'].append(seed2)
                    found_di['candidate'].append(candidate)
                    found_di['attempts'].append(attempts)
                    found_di['temperature'].append(temp)
                    if model_ID is not None:
                        found_di['rdkit qspr'].append(pre)
                    if verbose == 0:
                        if model_ID is not None:
                            print("rdkit qspr output:\t{}".format(pre))
                        print("cat seed:\t{}".format(seed1))
                        print("ani seed:\t{}".format(seed2))
                        print("candidate:\t{}".format(candidate))
                        print("attempts:\t{}".format(attempts))
                    found += 1
        if current_found < found: #did we find a soln this round
            if verbose == 1:
                clear_output(wait=True)
                if pd.DataFrame(found_di).shape[0] > 1:
                    print(pd.DataFrame(found_di).iloc[-1])
                elif pd.DataFrame(found_di).shape[0] == 1:
                    print(pd.DataFrame(found_di).iloc[0])
                print('{}/{} found'.format(found,find))
        if find <= found:
            return pd.DataFrame(found_di)
            break
        if attempts % 100 == 0:
            if verbose == 1:
                clear_output(wait=True)
                if pd.DataFrame(found_di).shape[0] > 1:
                    print(pd.DataFrame(found_di).iloc[-1])
                elif pd.DataFrame(found_di).shape[0] == 1:
                    print(pd.DataFrame(found_di).iloc[0])
                print('{}/{} found'.format(found,find))
                print('attempt {}'.format(attempts))