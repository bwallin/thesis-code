'''
Script to generate surface
'''
import pdb
import cPickle
from optparse import OptionParser
import logging

from pylab import *
from scipy import stats

from stats_util import Categorical

### Parse command line options
usage = "usage: %prog [options] model"
cmdline_parser = OptionParser(usage=usage)
cmdline_parser.add_option('--name', dest='name', metavar='FILE',
                          default='simulation',
                          help='Name for output file series')
cmdline_parser.add_option('-d', '--debug', dest='loglevel', default=logging.WARNING,
                          action='store_const', const=logging.DEBUG,
                          help='Debug level logging')
cmdline_parser.add_option('-n', '--length', dest='length',
                          type='int', default=1000,
                          help='Number of data points')
cmdline_parser.add_option('-v', '--visualize', dest='visualize',
                          action='store_true',
                          help='Visualize results')
options, args = cmdline_parser.parse_args()
param_module = args[0]

param_module = __import__(param_module)
logging.basicConfig(level=options.loglevel, format='%(asctime)s %(message)s')

n = options.length

dx = param_module.dx
mu_g_0 = param_module.mu_g_0
sigma_g_0 = param_module.sigma_g_0
mu_h_0 = param_module.mu_h_0
sigma_h_0 = param_module.sigma_h_0
phi = param_module.phi
mu_h = param_module.mu_h
transition_var_g = param_module.transition_var_g
observation_var_g = param_module.observation_var_g
transition_var_h = param_module.transition_var_h
observation_var_h = param_module.observation_var_h
noise_proportion = param_module.noise_proportion
canopy_cover = param_module.canopy_cover
cover_transition_matrix = param_module.cover_transition_matrix

transition_g_rv = stats.norm(0, sqrt(transition_var_g))
transition_h_rv = stats.norm(0, sqrt(transition_var_h))
observation_g_rv = stats.norm(0, observation_var_g)
observation_h_rv = stats.norm(0, observation_var_h)
noise_rv = stats.uniform(loc=-50, scale=100)

T = zeros((n,), dtype=int)
C = zeros((n,), dtype=int)
g = zeros((n,))
h = zeros((n,))

# Initialize
g[0] = stats.norm(mu_g_0, sigma_g_0).rvs()
h[0] = stats.norm(mu_h_0, sigma_h_0).rvs()
C[0] = Categorical((1/3., 1/3., 1/3.)).rvs()
p = array([noise_proportion,
           (1-canopy_cover[C[0]])*(1-noise_proportion),
           (canopy_cover[C[0]])*(1-noise_proportion)])
T[0] = Categorical(p).rvs()
if T[0] == 0:
    z = noise_rv.rvs()
if T[0] == 1:
    z = g[0] + observation_g_rv.rvs()
if T[0] == 2:
    z = g[0] + h[0] + observation_h_rv.rvs()

data = [[0, 0, z, 0, T[0]]]
for i in range(1, n):
    C[i] = Categorical(cover_transition_matrix[C[i-1],:]).rvs()
    p = array([noise_proportion,
               (1-canopy_cover[C[i]])*(1-noise_proportion),
               (canopy_cover[C[i]])*(1-noise_proportion)])
    T[i] = Categorical(p).rvs()
    g[i] = g[i-1] + transition_g_rv.rvs()
    h[i] = phi*(h[i-1] - mu_h) + mu_h + transition_h_rv.rvs()
    if T[i] == 0:
        z = noise_rv.rvs()
    elif T[i] == 1:
        z = g[i] + observation_g_rv.rvs()
    else:
        z = g[i] + h[i] + observation_h_rv.rvs()
    data += [[dx*i, 0, z, i, T[i]]]

if options.visualize:
    data=array(data)
    plot(data[:,0], data[:,2], '.')
    show()

with open('../data/%s.txt'%options.name, 'w') as f:
    for row in data:
        f.write(' '.join([str(el) for el in row])+'\n')

with open('../data/%s.pkl'%options.name, 'wb') as f:
    results = {'T': T,
               'C': C,
               'h': h,
               'g': g,
               'phi': phi,
               'mu_h': mu_h,
               'transition_var_g': transition_var_g,
               'observation_var_g': observation_var_g,
               'transition_var_h': transition_var_h,
               'observation_var_h': observation_var_h,
               'mu_g_0': mu_g_0,
               'sigma_g_0': sigma_g_0,
               'mu_h_0': mu_h_0,
               'sigma_h_0': sigma_h_0,
               'noise_proportion': noise_proportion,
               'canopy_cover': canopy_cover,
               'cover_transition_matrix': cover_transition_matrix}
    cPickle.dump(results, f)
