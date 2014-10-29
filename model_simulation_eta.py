'''
Noise, ground, canopy, cover, AND multiple observations per shot.
'''
from __future__ import division
import sys
import pdb
import pickle
import logging

from pylab import array, zeros, mean, ones, eye, sqrt
from scipy import stats, ma, exp, log, nan, isnan, inf
from scipy import isinf, logical_or, logical_and
from scipy.stats import bernoulli, beta
from sklearn.hmm import MultinomialHMM

p = sys.path
sys.path.insert(0, '/home/bruce/Dropbox/thesis/code/pykalman')
from pykalman import KalmanFilter
sys.path = p

from gibbs import Model, GibbsStep
from gibbs import raw_sample_handler, indep_meanvar_handler, discrete_handler
from misc import forward_filter_backward_sample
from stats_util import Dirichlet, Categorical, MVNormal, IID, Multinomial


###############################################################################
def define_model(params_module, data):
    # Builds model object 
    # Encapsulates everything the gibbs sampler needs to know about.

    # Parameters and initialization values
    n = len(list(set(data.shot_id)))
    N = len(data)
    known_params = params_module.get_known_params(data)
    initials = params_module.get_initials(data)
    hyper_params = params_module.get_hyper_params(data)
    m_cover = params_module.m_cover
    m_type = params_module.m_type


    # Variables to be sampled (in this order)
    variable_names = ['h', 'g', 'T', 'C', 'noise_proportion', 'transition_var_g', 'transition_var_h']

    priors = {'g': MVNormal(hyper_params['g']['mu'], hyper_params['g']['cov']),
              'h': MVNormal(hyper_params['h']['mu'], hyper_params['h']['cov']),
              'C': IID(Categorical(hyper_params['C']['p']), n),
              'T': IID(Categorical(hyper_params['T']['p']), N),
              'noise_proportion': beta(*hyper_params['noise_proportion']['alpha']),
              'transition_var_g': stats.invgamma(hyper_params['transition_var_g']['a'], scale=hyper_params['transition_var_g']['b']),
              'transition_var_h': stats.invgamma(hyper_params['transition_var_h']['a'], scale=hyper_params['transition_var_h']['b'])}
    FCP_samplers = {'g': ground_elev_step(),
                    'h': canopy_height_step(),
                    'C': cover_step(),
                    'T': type_step(),
                    'noise_proportion': noise_proportion_step(),
                    'transition_var_g': transition_var_g_step(),
                    'transition_var_h': transition_var_h_step()}
    sample_handlers = {'g': [indep_meanvar_handler()],
                       'h': [indep_meanvar_handler()],
                       'T': [discrete_handler(support=range(m_type), length=N)],
                       'C': [discrete_handler(support=range(m_cover), length=n)],
                       'noise_proportion': [raw_sample_handler()],
                       'transition_var_g': [raw_sample_handler()],
                       'transition_var_h': [raw_sample_handler()]}
    diagnostic_variable = 'noise_proportion'

    model = Model()
    model.set_variable_names(variable_names)
    model.set_known_params(known_params)
    model.set_hyper_params(hyper_params)
    model.set_priors(priors)
    model.set_initials(initials)
    model.set_FCP_samplers(FCP_samplers)
    model.set_sample_handlers(sample_handlers)
    model.set_diagnostic_variable(diagnostic_variable)
    model.set_data(data)

    return model


####################################################################################################
# Gibbs sampler parts - full conditional posterior/metropolis-hastings samplers
class ground_elev_step(GibbsStep):
    def __init__(self, *args, **kwargs):
        super(ground_elev_step, self).__init__(*args, **kwargs)
        self._kalman = KalmanFilter()
        self.counter = 10

    def sample(self, model, evidence):
        z = evidence['z']
        T = evidence['T']
        g = evidence['g']
        h = evidence['h']
        transition_var_g = evidence['transition_var_g']
        shot_id = evidence['shot_id']

        observation_var_g = model.known_params['observation_var_g']
        observation_var_h = model.known_params['observation_var_h']
        prior_mu_g = model.hyper_params['g']['mu'] 
        prior_cov_g = model.hyper_params['g']['cov'] 
        N = len(z)
        n = len(g)

        # Make g, h, and z vector valued to avoid ambiguity
        g = g.copy().reshape((n, 1))
        h = h.copy().reshape((n, 1))
        
        z_g = ma.asarray(nan + zeros((n, 1)))
        obs_cov = ma.asarray(inf + zeros((n, 1, 1)))
        for i in xrange(n):
            z_i = z[shot_id == i]
            T_i = T[shot_id == i]
            if 1 in T_i and 2 in T_i:
                # Sample mean and variance for multiple observations
                n_obs_g, n_obs_h = sum(T_i == 1), sum(T_i == 2)
                obs_cov_g, obs_cov_h = observation_var_g/n_obs_g, observation_var_h/n_obs_h
                z_g[i] = (mean(z_i[T_i == 1])/obs_cov_g + mean(z_i[T_i == 2] - h[i])/obs_cov_h)/(1/obs_cov_g + 1/obs_cov_h)
                obs_cov[i] = 1/(1/obs_cov_g + 1/obs_cov_h)
            elif 1 in T_i:
                n_obs_g = sum(T_i == 1) 
                z_g[i] = mean(z_i[T_i == 1])
                obs_cov[i] = observation_var_g/n_obs_g
            elif 2 in T_i:
                n_obs_h = sum(T_i == 2) 
                z_g[i] = mean(z_i[T_i == 2] - h[i])
                obs_cov[i] = observation_var_h/n_obs_h

        z_g[isnan(z_g)] = ma.masked
        obs_cov[isinf(obs_cov)] = ma.masked

        kalman = self._kalman
        kalman.initial_state_mean = array([prior_mu_g[0],])
        kalman.initial_state_covariance = array([prior_cov_g[0],])
        kalman.transition_matrices = eye(1)
        kalman.transition_covariance = array([transition_var_g,])
        kalman.observation_matrices = eye(1)
        kalman.observation_covariance = obs_cov
        sampled_g = forward_filter_backward_sample(kalman, z_g, prior_mu_g, prior_cov_g)
        return sampled_g.reshape((n,))


class canopy_height_step(GibbsStep):
    def __init__(self, *args, **kwargs):
        super(canopy_height_step, self).__init__(*args, **kwargs)
        self._kalman = KalmanFilter()

    def sample(self, model, evidence):
        z = evidence['z']
        g = evidence['g']
        h = evidence['h']
        T = evidence['T']
        phi  = evidence['phi']
        transition_var_h = evidence['transition_var_h']
        shot_id = evidence['shot_id']

        observation_var_h = model.known_params['observation_var_h']
        mu_h = model.known_params['mu_h']
        prior_mu_h = model.hyper_params['h']['mu']
        prior_cov_h = model.hyper_params['h']['cov']
        n = len(h)
        N = len(z)

        # Making g, h, and z vector valued to avoid ambiguity
        g = g.copy().reshape((n,1))
        h = h.copy().reshape((n,1))

        z_h = ma.asarray(nan + zeros((n, 1)))
        obs_cov = ma.asarray(inf + zeros((n, 1, 1)))
        for i in xrange(n):
            z_i = z[shot_id == i]
            T_i = T[shot_id == i]
            if 2 in T_i:
                # Sample mean and variance for multiple observations
                n_obs = sum(T_i == 2)
                z_h[i] = mean(z_i[T_i == 2])
                obs_cov[i] = observation_var_h/n_obs

        z_h[isnan(z_h)] = ma.masked
        obs_cov[isinf(obs_cov)] = ma.masked

        kalman = self._kalman
        kalman.initial_state_mean = array([prior_mu_h[0],])
        kalman.initial_state_covariance = array([prior_cov_h[0],])
        kalman.transition_matrices = array([phi,])
        kalman.transition_covariance = array([transition_var_h,])
        kalman.transition_offsets = mu_h*(1-phi)*ones((n, 1))
        kalman.observation_matrices = eye(1)
        kalman.observation_offsets = g
        kalman.observation_covariance = obs_cov
        sampled_h = forward_filter_backward_sample(kalman, z_h, prior_mu_h, prior_cov_h)

        return sampled_h.reshape((n,))


class transition_var_g_step(GibbsStep):
    def sample(self, model, evidence):
        g = evidence['g']

        prior_mu_g = model.hyper_params['g']['mu']
        a = model.hyper_params['transition_var_g']['a']
        b = model.hyper_params['transition_var_g']['b']
        max_var = model.hyper_params['transition_var_g']['max']

        n = len(g)

        g_var_posterior = stats.invgamma(a + (n-1)/2., scale=b + sum((g[1:] - g[:-1])**2)/2.)
        g_var = g_var_posterior.rvs()

        return min(g_var, max_var)


class transition_var_h_step(GibbsStep):
    def sample(self, model, evidence):
        h = evidence['h']

        prior_mu_h = model.hyper_params['h']['mu']
        a = model.hyper_params['transition_var_h']['a']
        b = model.hyper_params['transition_var_h']['b']
        max_var = model.hyper_params['transition_var_h']['max']
        
        phi = model.known_params['phi']
        mu = model.known_params['mu_h']

        n = len(h)

        h_var_posterior = stats.invgamma(a + (n-1)/2., scale=b + sum(((h[1:]-mu) - phi*(h[:-1]-mu))**2)/2.)
        h_var = h_var_posterior.rvs()

        return min(h_var, max_var)


class type_step(GibbsStep):
    def sample(self, model, evidence):
        g = evidence['g']
        h = evidence['h']
        C = evidence['C']
        z = evidence['z']
        shot_id = evidence['shot_id']
        noise_proportion = evidence['noise_proportion']
        observation_var_g = evidence['observation_var_g']
        observation_var_h = evidence['observation_var_h']

        canopy_cover = model.known_params['canopy_cover']
        z_min = model.known_params['z_min']
        z_max = model.known_params['z_max']

        prior_p = model.hyper_params['T']['p']

        N = len(z)
        T = zeros(N)
        noise_rv = stats.uniform(z_min, z_max - z_min)
        min_index = min(z.index)
        for i in shot_id.index:
            l = zeros(3)
            index = i-min_index
            shot_index = shot_id[i]-min(shot_id)
            l[0] = noise_proportion*noise_rv.pdf(z[i])
            g_norm = stats.norm(g[shot_index], sqrt(observation_var_g))
            C_i = canopy_cover[C[shot_index]]
            l[1] = (1-noise_proportion)*(1-C_i)*g_norm.pdf(z[i])
            h_norm = stats.norm(h[shot_index] + g[shot_index], sqrt(observation_var_h))
            if z[i] > g[shot_index]+3:
                l[2] = (1-noise_proportion)*(C_i)*h_norm.pdf(z[i])
            p = l/sum(l)
            T[index] = Categorical(p).rvs()

        return T


class cover_step(GibbsStep):
    def sample(self, model, evidence):
        noise_proportion = evidence['noise_proportion'] 
        T = evidence['T'] 
        C = evidence['C']
        shot_id = evidence['shot_id']

        canopy_cover = model.known_params['canopy_cover']
        cover_transition_matrix = model.known_params['cover_transition_matrix']

        n = len(C)
        m_type = 3
        m_cover = len(canopy_cover)

        emissions = array([[noise_proportion, 
                            (1-noise_proportion)*(1-canopy_cover[i]), 
                            (1-noise_proportion)*(canopy_cover[i])] for i in xrange(m_cover)])

        counts = [sum(T[shot_id == 0] == j) for j in range(m_type)]
        emission_likes = [Multinomial(emissions[j,:]).pmf(counts) for j in xrange(m_cover)]
        transition_likes = cover_transition_matrix[:,C[1]]
        C[0] = Categorical(emission_likes * transition_likes).rvs()
        for i in xrange(1, n-1):
            counts = [sum(T[shot_id == i] == j) for j in range(m_type)]
            emission_likes = [Multinomial(emissions[j,:]).pmf(counts) for j in xrange(m_cover)]
            transition_likes = cover_transition_matrix[C[i-1],:] * cover_transition_matrix[:,C[i+1]]
            C[i] = Categorical(emission_likes * transition_likes).rvs()
        counts = [sum(T[shot_id == (n-1)] == j) for j in range(m_type)]
        emission_likes = [Multinomial(emissions[j,:]).pmf(counts) for j in xrange(m_cover)]
        transition_likes = cover_transition_matrix[:,C[n-2]]
        C[n-1] = Categorical(emission_likes * transition_likes).rvs()

        return C


class noise_proportion_step(GibbsStep):
    def sample(self, model, evidence):
        T = evidence['T']

        alpha = model.hyper_params['noise_proportion']['alpha']

        N = len(T)
        n_noise = sum(T==0)
        counts = array((n_noise, N - n_noise))
        return Dirichlet(alpha + counts).rvs()[0]


def visualize_gibbs(sampler, evidence):
    pdb.set_trace()
    z, T, C, d, g, h, transition_var_g, transition_var_h, canopy_cover = \
            [evidence[var] for var in ['z', 'T', 'C', 'd', 'g', 'h', 'transition_var_g', 'transition_var_h', 'canopy_cover']]
    g = g.reshape((len(g), ))
    h = h.reshape((len(h), ))
    dists = sorted(list(set(d)))

    print "transition_var_g: %s" % transition_var_g
    print "transition_var_h: %s" % transition_var_h
    print "T counts: %s" % [sum(T==i) for i in range(3)]

    from matplotlib import pyplot as plt
    fig = plt.figure()
    plt.plot(d[T==0], z[T==0], 'r.')
    plt.plot(d[T==1], z[T==1], 'k.')
    plt.plot(d[T==2], z[T==2], 'g.')
    plt.plot(dists, g, 'k-', linewidth=3, alpha=.5)
    for i in xrange(len(canopy_cover)):
        canopy = ma.asarray(g+h)
        canopy[C!=i] = ma.masked
        plt.fill_between(dists, g, canopy, color='g', alpha=canopy_cover[i]*.7)
    def moveon(event):
        plt.close()
    fig.canvas.mpl_connect('key_press_event', moveon)
    plt.show()

