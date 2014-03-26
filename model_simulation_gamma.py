'''
No noise, ground and canopy model.
'''
from __future__ import division
import sys
import pdb
import pickle

from pylab import array, zeros, mean, ones, eye, sqrt
from pylab import isnan, logical_and
from scipy import stats, ma, exp, log, nan
from scipy.stats import bernoulli

#p = sys.path
#sys.path.insert(0, '/home/bruce/Dropbox/thesis/code/pykalman')
from pykalman import KalmanFilter
#sys.path = p

from gibbs import Model, GibbsStep
from misc import forward_filter_backward_sample
from stats_util import dirichlet, categorical, mvnorm, iid_dist

# Known values from simulation (use to initialize sampler for rapid convergence)
p = pickle.load(open('../data/sims/simulation_gamma.pkl', 'rb'))
g = p['g']
h = p['h']
T = p['T']
mu_h = p['mu_h']
phi = p['phi']
sigma_g = p['sigma_g']
sigma_h = p['sigma_h']
sigma_z_g = p['sigma_z_g']
sigma_z_h = p['sigma_z_h']
############################################################################################################
def define_model(data):
    # Builds model object
    m = 3
    n_points = len(data)
    n_shots = len(set(data['shot_id']))
    variable_names = ['g', 'h', 'T', 'p_type', 'sigma_g', 'sigma_h']
    known_params = {'sigma_z_g': sigma_z_g,
                    'sigma_z_h': sigma_z_h,
                    'mu_h': mu_h,
                    'phi': phi}
    hyper_params = {'alpha_type': array((0, 1., 1.)),
                    'prior_mu_g': -25.+zeros(n_shots),
                    'prior_cov_g': 100.*eye(n_shots),
                    'prior_mu_h': 30.+zeros(n_shots),
                    'prior_cov_h': 100.*eye(n_shots),
                    'a_g': 6,
                    'b_g': 1,
                    'a_h': 6,
                    'b_h': 1}
    initials = {}
    #initials = {'sigma_g': sigma_g,
    #            'sigma_h': sigma_h,
    #            'T': T[:n_shots],
    #            'g': g[:n_shots],
    #            'h': h[:n_shots]}
    priors = {'p_type': dirichlet(hyper_params['alpha_type']),
              'sigma_g': stats.invgamma(hyper_params['a_g'], scale=hyper_params['b_g']),
              'sigma_h': stats.invgamma(hyper_params['a_h'], scale=hyper_params['b_h']),
              'g': mvnorm(hyper_params['prior_mu_g'], hyper_params['prior_cov_g']),
              'h': mvnorm(hyper_params['prior_mu_h'], hyper_params['prior_cov_h']),
              'T': iid_dist(categorical(hyper_params['alpha_type']/sum(hyper_params['alpha_type'])), n_points)}
    FCP_samplers = {'p_type': p_type_step(),
                    'g': ground_height_step(),
                    'h': canopy_height_step(),
                    'sigma_g': sigma_ground_step(),
                    'sigma_h': sigma_height_step(),
                    'T': type_step()}

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
        z = evidence['z']
        T, g, sigma_g = [evidence[var] for var in ['T', 'g', 'sigma_g']]
        sigma_z_g = model.known_params['sigma_z_g']
        prior_mu_g, prior_cov_g = [model.hyper_params[var] for var in ['prior_mu_g', 'prior_cov_g']]
        n = len(g)

        z_g = ma.asarray(z.copy())
        z_g[T != 1] = nan # Necessary?
        z_g[isnan(z_g)] = ma.masked

        kalman = self._kalman
        kalman.initial_state_mean=prior_mu_g[0]
        kalman.initial_state_covariance=prior_cov_g[0,0]
        kalman.transition_matrices=1
        kalman.transition_covariance=sigma_g**2
        kalman.observation_matrices=1
        kalman.observation_covariance=sigma_z_g**2
        sampled_g = forward_filter_backward_sample(kalman, z_g)

        return sampled_g


class canopy_height_step(GibbsStep):
    def __init__(self, *args, **kwargs):
        super(canopy_height_step, self).__init__(*args, **kwargs)
        self._kalman = KalmanFilter()

    def sample(self, model, evidence):
        z, T, g, h, sigma_h, phi  = [evidence[var] for var in ['z', 'T', 'g', 'h', 'sigma_h', 'phi']]
        sigma_z_h = model.known_params['sigma_z_h']
        mu_h = model.known_params['mu_h']
        prior_mu_h = model.hyper_params['prior_mu_h']
        prior_cov_h = model.hyper_params['prior_cov_h']
        n = len(h)

        z_h = ma.asarray(z.copy())
        z_h[T != 2] = nan
        z_h[isnan(z_h)] = ma.masked

        kalman = self._kalman
        kalman.initial_state_mean = prior_mu_h[0]
        kalman.initial_state_covariance=prior_cov_h[0,0]
        kalman.transition_matrices=phi
        kalman.transition_covariance=sigma_h**2
        kalman.transition_offsets=mu_h*(1-phi)*ones((n, 1))
        kalman.observation_matrices=1
        kalman.observation_covariance=sigma_z_h**2
        kalman.observation_offsets=g
        sampled_h = forward_filter_backward_sample(kalman, z_h)

        return sampled_h


class sigma_ground_step(GibbsStep):
    def sample(self, model, evidence):
        prior_mu_g, a_g, b_g = [model.hyper_params[var] for var in ['prior_mu_g', 'a_g', 'b_g']]
        g = evidence['g']
        n = len(g)

        g_var_posterior = stats.invgamma(a_g + (n-1)/2., scale=b_g + sum((g[1:] - g[:-1])**2)/2.)
        g_var = g_var_posterior.rvs()

        return sqrt(g_var)


class sigma_height_step(GibbsStep):
    def sample(self, model, evidence):
        prior_mu_h, a_h, b_h = [model.hyper_params[var] for var in ['prior_mu_h', 'a_h', 'b_h']]
        h = evidence['h']
        n = len(h)

        h_var_posterior = stats.invgamma(a_h + (n-1)/2., scale=b_h + sum((h[1:] - h[:-1])**2)/2.)
        h_var = h_var_posterior.rvs()
        return sqrt(h_var)


class type_step(GibbsStep):
    def sample(self, model, evidence):
        g, h, p_type, z, sigma_z_g, sigma_z_h = [evidence[var]
                for var in ['g', 'h', 'p_type', 'z', 'sigma_z_g', 'sigma_z_h']]
        N = len(z)
        m = len(p_type)
        T = zeros(N)
        noise_rv = stats.uniform(z.min(), z.max() - z.min())
        for i in xrange(N):
            l = zeros(m)
            l[0] = p_type[0]*noise_rv.pdf(z[i])
            g_norm = stats.norm(g[i], sigma_z_g)
            l[1] = p_type[1]*g_norm.pdf(z[i])
            if z[i] > g[i]:
                h_norm = stats.norm(h[i], sigma_z_h)
                l[2] = p_type[2]*h_norm.pdf(z[i] - g[i])
            else:
                l[1] = 1
            p = l/sum(l)
            type_rv = categorical(p)
            T[i] = type_rv.rvs()

        return T


class p_type_step(GibbsStep):
    def sample(self, model, evidence):
        alpha_type = model.hyper_params['alpha_type']
        T = evidence['T']
        m = len(alpha_type)
        return dirichlet(alpha_type + array([sum(T==i) for i in xrange(m)])).rvs()


def visualize_gibbs(evidence):
    z, T, d, g, h, sigma_g = [evidence[var] for var in ['z', 'T', 'd', 'g', 'h', 'sigma_g']]
    g = g.reshape((len(g), ))
    h = h.reshape((len(h), ))
    from matplotlib import pyplot as plt
    fig = plt.figure()
    plt.plot(d[T==0], z[T==0], 'r.')
    plt.plot(d[T==1], z[T==1], 'k.')
    plt.plot(d[T==2], z[T==2], 'g.')
    plt.plot(d, g, 'k-', linewidth=3, alpha=.5)
    plt.plot(d, g+h, 'g-', linewidth=3, alpha=.5)
    #plt.fill_between(d, g-1.96*sigma_g, g+1.96*sigma_g, color='k', alpha=.3)
    def moveon(event):
        plt.close()
    fig.canvas.mpl_connect('key_press_event', moveon)
    plt.show()

