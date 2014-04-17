# Parameters and initialization values for Cedar4 data
import os
import pickle

from scipy import array, ones, eye

m_type = 3
m_cover = 5

def get_known_params(data):
    validation_filepath = os.path.splitext(data.filepath)[0]+'.pkl'
    validation_pickle = pickle.load(open(validation_filepath, 'r'))
    known_params = {'observation_var_g': validation_pickle['observation_var_g'], # observation stdev
                    'observation_var_h': validation_pickle['observation_var_h'],
                    'mu_h': validation_pickle['mu_h'], # mean canopy
                    'phi': validation_pickle['phi'], # canopy autoreg param
                    'z_min': -50,
                    'z_max': 50,
                    'canopy_cover': validation_pickle['canopy_cover'], # possible canopy cover states
                    'cover_transition_matrix': validation_pickle['cover_transition_matrix']} # canopy transition matrix
    return known_params

def get_initials(data):
    N = len(data)
    n = len(set(data.shot_id))
    validation_filepath = os.path.splitext(data.filepath)[0]+'.pkl'
    validation_pickle = pickle.load(open(validation_filepath, 'r'))
    initials = {'g': validation_pickle['g'],
                'h': validation_pickle['h'],
                'T': validation_pickle['T'],
                'C': validation_pickle['C'],
                'noise_proportion': validation_pickle['noise_proportion'],
                'transition_var_g': validation_pickle['transition_var_g'],
                'transition_var_h': validation_pickle['transition_var_h']}
    return initials

def get_hyper_params(data):
    N = len(data)
    n = len(set(data.shot_id))
    validation_filepath = os.path.splitext(data.filepath)[0]+'.pkl'
    validation_pickle = pickle.load(open(validation_filepath, 'r'))
    hyper_params = {'g': {'mu': validation_pickle['g'], 'cov': 1000*eye(n)}, # mvn prior
                    'h': {'mu': validation_pickle['h'], 'cov': 1000*eye(n)}, # mvn prior
                    'T': {'p': ones(m_type)/m_type}, # iid categorical prior
                    'C': {'p': ones(m_cover)/m_cover}, # iid categorical prior
                    'noise_proportion': {'alpha': array((0, 0))}, # dirichlet prior
                    'transition_var_g': {'a': 101, 'b': validation_pickle['transition_var_g']*100, 'max': 1}, # inv-gamma prior - mean = true val
                    'transition_var_h': {'a': 101, 'b': validation_pickle['transition_var_h']*100, 'max': 5}} # inv-gamma prior - mean = 
    return hyper_params

