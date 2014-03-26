'''
No noise, ground and canopy model.
'''
from __future__ import division
import sys
import pdb
import pickle

from pylab import array, zeros, mean, ones, eye, sqrt
from pylab import isnan, logical_and, ceil
from scipy import stats, ma, exp, log, std
from scipy.stats import bernoulli

#p = sys.path
#sys.path.insert(0, '/home/bruce/Dropbox/thesis/code/pykalman')
from pykalman import KalmanFilter
#sys.path = p

from gibbs import Model, GibbsStep
from misc import forward_filter_backward_sample
from stats_util import dirichlet, categorical, mvnorm, iid_dist

# Known g from simulation (use to initialize sampler)
p = pickle.load(open('../data/sims/simulation_alpha.pkl', 'rb'))
g = p['g']
sigma_g = p['sigma_g']
sigma_z_g = p['sigma_z_g']
############################################################################################################
def define_model(data):
    # Builds model object
    n = len(data)
    variable_names = ['g', 'sigma_g']
    known_params = {'sigma_z_g': sigma_z_g,
                    'T': ones(n)}
    hyper_params = {'prior_mu_g': 0+zeros(n),
                    'prior_cov_g': 100*eye(n),
                    'a_g': 0.,
                    'b_g': 0.}
    priors = {'sigma_g': stats.invgamma(hyper_params['a_g'], scale=hyper_params['b_g']),
              'g': mvnorm(hyper_params['prior_mu_g'], hyper_params['prior_cov_g'])}
    initials = {'g': g[:n],
                'sigma_g': sigma_g}
    FCP_samplers = {'g': ground_height_step(),
                    'sigma_g': sigma_ground_step()}

    model = Model()
    model.set_variable_names(variable_names)
    model.set_known_params(known_params)
    model.set_hyper_params(hyper_params)
    model.set_priors(priors)
    model.set_initials(initials)
    model.set_FCP_samplers(FCP_samplers)
    model.set_data(data)

    return model


####################################################################################################
# Gibbs sampler functions
class ground_height_step(GibbsStep):
    def __init__(self, *args, **kwargs):
        super(ground_height_step, self).__init__(*args, **kwargs)
        self._kalman = KalmanFilter()

    def sample(self, model, evidence):
        z, shot_id = evidence['z'], evidence['shot_id']
        g, sigma_g = [evidence[var] for var in ['g', 'sigma_g']]
        sigma_z_g = model.known_params['sigma_z_g']
        mu_g, cov_g = [model.hyper_params[var] for var in ['prior_mu_g', 'prior_cov_g']]

        kalman = self._kalman
        kalman.initial_state_mean = mu_g[0]
        kalman.initial_state_covariance = cov_g[0,0]
        kalman.transition_matrices = 1
        kalman.transition_covariance = sigma_g**2
        kalman.observation_matrices = 1
        kalman.observation_covariance = sigma_z_g**2
        sampled_g = forward_filter_backward_sample(kalman, z)

        return sampled_g


class sigma_ground_step(GibbsStep):
    def sample(self, model, evidence):
        prior_mu_g, a_g, b_g = [model.hyper_params[var] for var in ['prior_mu_g', 'a_g', 'b_g']]
        g = evidence['g']
        n = len(g)

        g_var_posterior = stats.invgamma(a_g + (n-1)/2., scale=b_g + sum((g[1:] - g[:-1])**2)/2.)
        g_var = g_var_posterior.rvs()
        return sqrt(g_var)


def visualize_priors(priors):
    from matplotlib import pyplot as plt
    N = len(priors)
    m = round(sqrt(N*3/4.))
    n = ceil(N/m)
    fig = plt.figure()
    for i, var in enumerate(priors):
        ax = fig.add_subplot(n,m,i)
        ax = ax.hist(priors[var].rvs(1000), 20)
    plt.show()


def visualize_gibbs(evidence):
    z, d, g, sigma_g = [evidence[var] for var in ['z', 'd', 'g', 'sigma_g']]
    g = g.reshape((len(g), ))
    from matplotlib import pyplot as plt
    fig = plt.figure()
    plt.plot(d, z, 'k.')
    plt.plot(d, g, 'g-', alpha=.5, linewidth=3)
    #plt.fill_between(d, g-1.96*sigma_g, g+1.96*sigma_g, color='k', alpha=.3)
    def moveon(event):
        plt.close()
    fig.canvas.mpl_connect('key_press_event', moveon)
    plt.show()

