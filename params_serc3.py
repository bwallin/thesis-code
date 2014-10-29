# Parameters and initialization values for Cedar4 data
from scipy import array, ones, eye

from zeta_params import *

phi = .95
mu_h = 30
observation_var_g = 1 
observation_var_h = 16 
noise_proportion = .37
canopy_cover = array([0,.25, .5, .75]) 
cover_transition_matrix = array([[.99, .0033, .0033, .0033], 
                                 [.0033, .99, .0033, .0033], 
                                 [.0033, .0033, .99, .0033],
                                 [.0033, .0033, .0033, .99]])
m_cover = len(canopy_cover)
m_type = 3
z_min, z_max = -50, 50
g_guess = -25

def get_known_params(data):
    known_params = {'observation_var_g': observation_var_g, # observation stdev
                    'observation_var_h': observation_var_h,
                    'mu_h': mu_h, # mean canopy
                    'phi': phi, # canopy autoreg param
                    'z_min': z_min,
                    'z_max': z_max,
                    'canopy_cover': canopy_cover, # possible canopy cover states
                    'cover_transition_matrix': cover_transition_matrix} # canopy transition matrix
    return known_params

def get_initials(data):
    n = len(set(data.shot_id))
    N = len(data)
    initials = {'g': g_guess*ones(n),
                'h': 2*mu_h*ones(n),
                'T': (abs(array(data.z - g_guess)) < 2),
                'noise_proportion': noise_proportion,
                'transition_var_g': .1,
                'transition_var_h': 1}
    return initials

def get_hyper_params(data):
    n = len(set(data.shot_id))
    N = len(data)
    hyper_params = {'g': {'mu': -30*ones(n), 'cov': 1000*ones(n)}, # iid normal prior
                    'h': {'mu': mu_h*ones(n), 'cov': 1000*ones(n)}, # iid normal prior
                    'T': {'p': ones(m_type)/m_type}, # iid categorical prior
                    'C': {'p': ones(m_cover)/m_cover}, # iid categorical prior
                    'noise_proportion': {'alpha': array((N*.3, N*.3))}, # dirichlet prior
                    'transition_var_g': {'a': 1001, 'b': 100, 'max': 2}, # inv-gamma prior - mean = .1
                    'transition_var_h': {'a': 1001, 'b': 2000, 'max': 5}} # inv-gamma prior - mean = 2 
    return hyper_params

