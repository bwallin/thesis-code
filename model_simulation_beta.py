'''
No noise, ground and canopy model.
'''
from __future__ import division
import sys
import pdb
import pickle

from pylab import array, zeros, mean, ones, eye, sqrt
from pylab import isnan, logical_and
from scipy import stats, ma, exp, log, std, nan
from scipy.stats import bernoulli

#p = sys.path
#sys.path.insert(0, '/home/bruce/Dropbox/thesis/code/pykalman')
from pykalman import KalmanFilter
#sys.path = p

from gibbs import Model, GibbsStep
from misc import forward_filter_backward_sample
from stats_util import dirichlet, categorical, mvnorm, iid_dist

# Known g from simulation (use to initialize sampler)
#p = pickle.load(open('../data/sims/simulation_beta_1.pkl', 'rb'))
#g = p['g']
#sigma_g = p['sigma_g']
#sigma_z_g = p['sigma_z_g']
sigma_z_g = .3
############################################################################################################
def define_model(data):
    # Builds model object
    n = len(data)
    variable_names = ['g', 'sigma_g', 'p_type', 'T']
    known_params = {'sigma_z_g': sigma_z_g}
    hyper_params = {'prior_mu_g': 0*ones(n),
                    'prior_cov_g': 100*eye(n),
                    'alpha_type': (1., 1.),
                    'a_g': 3.,
                    'b_g': 1.}
    priors = {'sigma_g': stats.invgamma(hyper_params['a_g'], scale=hyper_params['b_g']),
              'p_type': dirichlet(hyper_params['alpha_type']),
              'T': iid_dist(categorical((1., 1.)), n),
              'g': mvnorm(hyper_params['prior_mu_g'], hyper_params['prior_cov_g'])}
    #initials = {'g': g[:n],
    #            'sigma_g': sigma_g}
    FCP_samplers = {'g': ground_height_step(),
                    'p_type': p_type_step(),
                    'T': type_step(),
                    'sigma_g': sigma_ground_step()}

    model = Model()
    model.set_variable_names(variable_names)
    model.set_known_params(known_params)
    model.set_hyper_params(hyper_params)
    model.set_priors(priors)
    #model.set_initials(initials)
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

        z_g = z.copy()
        z_g[T == 0] = nan
        z_g = ma.asarray(z_g)
        z_g[isnan(z_g)] = ma.masked

        kalman = self._kalman
        kalman.initial_state_mean=prior_mu_g[0]
        kalman.initial_state_covariance=prior_cov_g[0,0]
        kalman.transition_matrices=1
        kalman.transition_covariance=sigma_g**2
        kalman.observation_matrices=1
        kalman.observation_covariance=sigma_z_g**2
        pdb.set_trace()
        sampled_g = forward_filter_backward_sample(kalman, z_g)

        return sampled_g


class sigma_ground_step(GibbsStep):
    def sample(self, model, evidence):
        prior_mu_g, a_g, b_g = [model.hyper_params[var] for var in ['prior_mu_g', 'a_g', 'b_g']]
        g = evidence['g']
        n = len(g)

        g_var_posterior = stats.invgamma(a_g + (n-1)/2., scale=b_g + sum((g[1:] - g[:-1])**2)/2.)
        g_var = g_var_posterior.rvs()

        return sqrt(g_var)


class type_step(GibbsStep):
    def sample(self, model, evidence):
        g, p_type, z, sigma_z_g = [evidence[var]
                for var in ['g', 'p_type', 'z', 'sigma_z_g']]
        n = len(g)
        m = len(p_type)
        T = zeros(n)
        noise_rv = stats.uniform(z.min(), z.max() - z.min())
        for i in xrange(n):
            l = zeros(m)
            l[0] = p_type[0]*noise_rv.pdf(z[i])
            g_norm = stats.norm(g[i], sigma_z_g)
            l[1] = p_type[1]*g_norm.pdf(z[i])
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
    z, T, d, g, sigma_g = [evidence[var] for var in ['z', 'T', 'd', 'g', 'sigma_g']]
    d_shot = sorted(list(set(d)))
    g = g.reshape((len(g), ))
    from matplotlib import pyplot as plt
    fig = plt.figure()
    plt.plot(d[T==0], z[T==0], 'r.')
    plt.plot(d[T==1], z[T==1], 'k.')
    plt.plot(d_shot, g, 'k')
    plt.fill_between(d_shot, g-1.96*sigma_g, g+1.96*sigma_g, color='k', alpha=.3)
    def moveon(event):
        plt.close()
    fig.canvas.mpl_connect('key_press_event', moveon)
    plt.show()

