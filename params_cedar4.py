# Parameters and initialization values for Cedar4 data
from scipy import array, ones, eye

phi = .95
mu_h = 10
observation_var_g = 1 
observation_var_h = 9 
noise_proportion = .36
canopy_cover = array([0,.25, .5, .75, 1.]) 
cover_transition_matrix = array([[.99, .0025, .0025, .0025, .0025], 
                                 [.0025, .99, .0025, .0025, .0025], 
                                 [.0025, .0025, .99, .0025, .0025],
                                 [.0025, .0025, .0025, .99, .0025],
                                 [.0025, .0025, .0025, .0025, .99]])
m_cover = 5
m_type = 3
z_min, z_max = -50, 50

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
    N = len(data)
    n = len(set(data.shot_id))
    g_guess = 15
    initials = {'g': g_guess*ones(n),
                'h': mu_h*2*ones(n),
                'T': abs(data.z - g_guess) < 2,
                'C': 2*ones(n, dtype=int),
                'noise_proportion': .36,
                'transition_var_g': .1,
                'transition_var_h': 1}
    return initials

def get_hyper_params(data):
    N = len(data)
    n = len(set(data.shot_id))
    hyper_params = {'g': {'mu': 0*ones(n), 'cov': 1000*eye(n)}, # mvn prior
                    'h': {'mu': mu_h*ones(n), 'cov': 1000*eye(n)}, # mvn prior
                    'T': {'p': ones(m_type)/m_type}, # iid categorical prior
                    'C': {'p': ones(m_cover)/m_cover}, # iid categorical prior
                    'noise_proportion': {'alpha': array((N*.3, N*.3))}, # dirichlet prior
                    'transition_var_g': {'a': 11, 'b': 1, 'max': 1}, # inv-gamma prior - mean = .1
                    'transition_var_h': {'a': 101, 'b': 200, 'max': 5}} # inv-gamma prior - mean = 2 
    return hyper_params

