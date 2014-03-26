'''
Script to generate surface
'''
import pdb
import cPickle
from optparse import OptionParser
import logging

from pylab import *
from scipy import stats

from stats_util import categorical

### Parse command line options
usage = "usage: %prog [options]"
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
                          help='Visualize intermediate results')
cmdline_parser.add_option('--phi', dest='phi',
                          type='float', default=0,
                          help='Decay coefficient for canopy')
cmdline_parser.add_option('--mu-h', dest='mu_h',
                          type='float', default=0,
                          help='Mean canopy height')
cmdline_parser.add_option('--mu-g-0', dest='mu_g_0',
                          type='float', default=0,
                          help='Mean initial ground height')
cmdline_parser.add_option('--sigma-g-0', dest='sigma_g_0',
                          type='float', default=1,
                          help='Initial ground stdev')
cmdline_parser.add_option('--mu-h-0', dest='mu_h_0',
                          type='float', default=0,
                          help='Mean initial canopy height')
cmdline_parser.add_option('--sigma-h-0', dest='sigma_h_0',
                          type='float', default=1,
                          help='Initial canopy stdev')
cmdline_parser.add_option('--sigma-g', dest='sigma_g',
                          type='float', default=1,
                          help='Transition stdev of ground')
cmdline_parser.add_option('--sigma-z-g', dest='sigma_z_g',
                          type='float', default=1,
                          help='Observation stdev of ground')
cmdline_parser.add_option('--sigma-h', dest='sigma_h',
                          type='float', default=1,
                          help='Transition stdev of canopy height')
cmdline_parser.add_option('--sigma-z-h', dest='sigma_z_h',
                          type='float', default=1,
                          help='Observation stdev of canopy height')
cmdline_parser.add_option('-p', '--noise-proportion', dest='noise_proportion',
                          type='float', default=.5,
                          help='Noise proportion')
cmdline_parser.add_option('-q', '--ground-proportion', dest='ground_proportion',
                          type='float', default=.5,
                          help='Ground proportion')
options, args = cmdline_parser.parse_args()

logging.basicConfig(level=options.loglevel, format='%(asctime)s %(message)s')

n = options.length
dx = .7

mu_g_0 = options.mu_g_0
sigma_g_0 = options.sigma_g_0
mu_h_0 = options.mu_h_0
sigma_h_0 = options.sigma_h_0

phi = options.phi
mu_h = options.mu_h
sigma_g = options.sigma_g
sigma_z_g = options.sigma_z_g
sigma_h = options.sigma_h
sigma_z_h = options.sigma_z_h

p = options.noise_proportion
q = options.ground_proportion
r = 1-p-q
proportions = (p, q, r)

transition_g_rv = stats.norm(0, sigma_g)
transition_h_rv = stats.norm(0, sigma_h)
observation_g_rv = stats.norm(0, sigma_z_g)
observation_h_rv = stats.norm(0, sigma_z_h)
type_rv = categorical(proportions)
noise_rv = stats.uniform(loc=-40, scale=80)

t = zeros((n,), dtype=int)
g = zeros((n,))
h = zeros((n,))
g[0] = stats.norm(mu_g_0, sigma_g_0).rvs()
h[0] = stats.norm(mu_h_0, sigma_h_0).rvs()
t[0] = type_rv.rvs()[0]
if t[0] == 0:
    z = noise_rv.rvs()
if t[0] == 1:
    z = g[0] + observation_g_rv.rvs()
if t[0] == 2:
    z = g[0] + h[0] + observation_h_rv.rvs()

data = [[0, 0, z, 0, t[0]]]
for i in range(1, n):
    t[i] = type_rv.rvs()[0]
    g[i] = g[i-1] + transition_g_rv.rvs()
    h[i] = phi*(h[i-1] - mu_h) + mu_h + transition_h_rv.rvs()
    if t[i] == 0:
        z = noise_rv.rvs()
    elif t[i] == 1:
        z = g[i] + observation_g_rv.rvs()
    else:
        z = g[i] + h[i] + observation_h_rv.rvs()
    data += [[dx*i, 0, z, i, t[i]]]

data=array(data)
plot(data[:,0], data[:,2], '.')

show()

with open('../data/%s.txt'%options.name, 'w') as f:
    for row in data:
        f.write(' '.join([str(el) for el in row])+'\n')

with open('../data/%s.pkl'%options.name, 'wb') as f:
    results = {'t': t,
               'h': h,
               'g': g,
               'phi': phi,
               'mu_h': mu_h,
               'sigma_g': sigma_g,
               'sigma_z_g': sigma_z_g,
               'sigma_h': sigma_h,
               'sigma_z_h': sigma_z_h,
               'mu_g_0': mu_g_0,
               'sigma_g_0': sigma_g_0,
               'mu_h_0': mu_h_0,
               'sigma_h_0': sigma_h_0,
               'proportions': proportions}
    cPickle.dump(results, f)
