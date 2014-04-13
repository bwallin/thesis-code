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
from sklearn.hmm import MultinomialHMM

p = sys.path
sys.path.insert(0, '/home/bruce/Dropbox/thesis/code/pykalman')
from pykalman import KalmanFilter
sys.path = p

from gibbs import Model, GibbsStep
from misc import forward_filter_backward_sample
from stats_util import dirichlet, categorical, mvnorm, iid_dist

############################################################################################################
def define_model(data):
    # Builds model object
    # Known values from simulation (can use to initialize sampler/skip burn-in)
    p = pickle.load(open('../data/sims/simulation_zeta_3.pkl', 'rb'))
    g = p['g']
    h = p['h']
    T = p['T']
    C = p['C']
    sigma_g = p['sigma_g']
    sigma_h = p['sigma_h']
    noise_proportion = p['noise_proportion']
    # Also needed for known parameters (so load the right pkl dummy!)
    phi = p['phi']
    mu_h = p['mu_h']
    sigma_z_g = p['sigma_z_g']
    sigma_z_h = p['sigma_z_h']
    canopy_cover = p['canopy_cover']
    cover_transition_matrix = p['cover_transition_matrix']

    n = len(data)
    z = data.get('z')
    variable_names = ['g', 'h', 'T', 'C', 'noise_proportion', 'sigma_g', 'sigma_h']
    known_params = {'sigma_z_g': sigma_z_g,
                    'sigma_z_h': sigma_z_h,
                    'mu_h': mu_h,
                    'phi': phi,
                    'canopy_cover': canopy_cover,
                    'cover_transition_matrix': cover_transition_matrix}
    hyper_params = {'alpha_noise': array((1.,1.)),
                    'alpha_T': array((1., 1., 1.)), # not used yet
                    'alpha_C': array((1., 1., 1.)), # not used yet
                    'prior_mu_g': -25.+zeros(n), # not fully used yet
                    'prior_cov_g': 100.*eye(n), # not fully used yet
                    'prior_mu_h': 30.+zeros(n), # not fully used yet
                    'prior_cov_h': 100.*eye(n), # not fully used yet
                    'a_g': 11,
                    'b_g': .1,
                    'a_h': 11,
                    'b_h': 40}
    initials = {'g': -25+zeros(n),
                'noise_proportion': .2,
                'sigma_g': .1,
                'sigma_h': 1,
                'T': array([(0 if abs(z[i]+25)>1 else 1) for i in xrange(n)]),
                'C': zeros(n, dtype=int)}
    #initials = {'sigma_g': sigma_g,
    #            'sigma_h': sigma_h,
    #            'noise_proportion': noise_proportion,
    #            'C': C[:n],
    #            'T': T[:n],
    #            'g': g[:n],
    #            'h': h[:n]}
    priors = {'noise_proportion': dirichlet(hyper_params['alpha_noise']),
              'sigma_g': stats.invgamma(hyper_params['a_g'], scale=hyper_params['b_g']),
              'sigma_h': stats.invgamma(hyper_params['a_h'], scale=hyper_params['b_h']),
              'g': mvnorm(hyper_params['prior_mu_g'], hyper_params['prior_cov_g']),
              'h': mvnorm(hyper_params['prior_mu_h'], hyper_params['prior_cov_h']),
              'C': iid_dist(categorical(hyper_params['alpha_C']), n),
              'T': iid_dist(categorical(hyper_params['alpha_T']), n)}
    FCP_samplers = {'noise_proportion': noise_proportion_step(),
                    'g': ground_height_step(),
                    'h': canopy_height_step(),
                    'sigma_g': sigma_ground_step(),
                    'sigma_h': sigma_height_step(),
                    'C': cover_step(),
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
# Gibbs sampler parts - full conditional posterior/metropolis-hastings samplers
class ground_height_step(GibbsStep):
    def __init__(self, *args, **kwargs):
        super(ground_height_step, self).__init__(*args, **kwargs)
        self._kalman = KalmanFilter()

    def sample(self, model, evidence):
        z = evidence['z']
        T, g, h, sigma_g = [evidence[var] for var in ['T', 'g', 'h', 'sigma_g']]
        sigma_z_g = model.known_params['sigma_z_g']
        sigma_z_h = model.known_params['sigma_z_h']
        prior_mu_g, prior_cov_g = [model.hyper_params[var] for var in ['prior_mu_g', 'prior_cov_g']]
        n = len(g)

        # Must be a more concise way to deal with scalar vs vector
        g = g.copy().reshape((n,1))
        h = h.copy().reshape((n,1))
        z_g = ma.asarray(z.copy().reshape((n,1)))
        obs_cov = sigma_z_g**2*ones((n,1,1))
        if sum(T == 0) > 0:
            z_g[T == 0] = nan
        if sum(T == 2) > 0:
            z_g[T == 2] -= h[T == 2]
            obs_cov[T == 2] = sigma_z_h**2
        z_g[isnan(z_g)] = ma.masked

        kalman = self._kalman
        kalman.initial_state_mean = array([prior_mu_g[0],])
        kalman.initial_state_covariance = array([prior_cov_g[0,0],])
        kalman.transition_matrices = eye(1)
        kalman.transition_covariance = array([sigma_g**2,])
        kalman.observation_matrices = eye(1)
        kalman.observation_covariance = obs_cov
        sampled_g = forward_filter_backward_sample(kalman, z_g)

        return sampled_g.reshape((n,))


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

        g = g.copy().reshape((n,1))
        h = h.copy().reshape((n,1))
        z_h = ma.asarray(z.copy().reshape((n,1)))
        if sum(T == 0) > 0:
            z_h[T == 0] = nan
        if sum(T == 1) > 0:
            z_h[T == 1] = nan
        if sum(T == 2) > 0:
            z_h[T == 2] -= g[T == 2]
        z_h[isnan(z_h)] = ma.masked

        kalman = self._kalman
        kalman.initial_state_mean = array([prior_mu_h[0],])
        kalman.initial_state_covariance = array([prior_cov_h[0,0],])
        kalman.transition_matrices = array([phi,])
        kalman.transition_covariance = array([sigma_h**2,])
        kalman.transition_offsets = mu_h*(1-phi)*ones((n, 1))
        kalman.observation_matrices = eye(1)
        kalman.observation_covariance = array([sigma_z_h**2,])
        sampled_h = forward_filter_backward_sample(kalman, z_h)

        return sampled_h.reshape((n,))


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
        return min(sqrt(h_var), 3)


class type_step(GibbsStep):
    def sample(self, model, evidence):
        g, h, noise_proportion, z, sigma_z_g, sigma_z_h = [evidence[var]
                for var in ['g', 'h', 'noise_proportion', 'z', 'sigma_z_g', 'sigma_z_h']]
        C = evidence['C']
        canopy_cover = model.known_params['canopy_cover']
        N = len(z)
        m = 3
        T = zeros(N)
        noise_rv = stats.uniform(z.min(), z.max() - z.min())
        for i in xrange(N):
            l = zeros(m)
            l[0] = noise_proportion*noise_rv.pdf(z[i])
            g_norm = stats.norm(g[i], sigma_z_g)
            l[1] = (1-noise_proportion)*(1-canopy_cover[C[i]])*g_norm.pdf(z[i])
            h_norm = stats.norm(h[i], sigma_z_h)
            l[2] = (1-noise_proportion)*(canopy_cover[C[i]])*h_norm.pdf(z[i] - g[i])
            p = l/sum(l)
            T[i] = categorical(p).rvs()

        return T


class cover_step(GibbsStep):
    def sample(self, model, evidence):
        noise_proportion, T, C = [evidence[var] for var in ['noise_proportion', 'T', 'C']]
        canopy_cover, cover_transition_matrix = [model.known_params[var] for var in ['canopy_cover', 'cover_transition_matrix']]
        n = len(T)
        m = len(canopy_cover)
        emissions = array([[noise_proportion, (1-noise_proportion)*(1-canopy_cover[i]), (1-noise_proportion)*(canopy_cover[i])] for i in xrange(m)])
        C[0] = categorical(cover_transition_matrix[:,C[1]]*emissions[:,T[0]]).rvs()
        for i in xrange(1, n-1):
            C[i] = categorical(cover_transition_matrix[C[i-1],:]*cover_transition_matrix[:,C[i+1]]*emissions[:,T[i]]).rvs()
        C[-1] = categorical(cover_transition_matrix[C[-2],:]*emissions[:,T[-1]]).rvs()

        return C


class noise_proportion_step(GibbsStep):
    def sample(self, model, evidence):
        alpha_noise = model.hyper_params['alpha_noise']
        T = evidence['T']
        n = len(T)
        n_noise = sum(T==0)
        return dirichlet(alpha_noise + (n_noise, n-n_noise)).rvs()[0]


def visualize_gibbs(sampler, evidence):
    pdb.set_trace()
    z, T, C, d, g, h, sigma_g, sigma_h, canopy_cover = [evidence[var] for var in ['z', 'T', 'C', 'd', 'g', 'h', 'sigma_g', 'sigma_h', 'canopy_cover']]
    g = g.reshape((len(g), ))
    h = h.reshape((len(h), ))

    print "sigma_g: %s" % sigma_g
    print "sigma_h: %s" % sigma_h
    print "T counts: %s" % [sum(T==i) for i in range(3)]

    from matplotlib import pyplot as plt
    fig = plt.figure()
    plt.plot(d[T==0], z[T==0], 'r.')
    plt.plot(d[T==1], z[T==1], 'k.')
    plt.plot(d[T==2], z[T==2], 'g.')
    plt.plot(d, g, 'k-', linewidth=3, alpha=.5)
    for i in xrange(len(canopy_cover)):
        canopy = ma.asarray(g+h)
        canopy[C!=i] = ma.masked
        plt.plot(d, canopy, 'g-', linewidth=3, alpha=canopy_cover[i]*.5)
    #plt.fill_between(d, g, g+h, color='g', alpha=array(canopy_cover)[C])
    def moveon(event):
        plt.close()
    fig.canvas.mpl_connect('key_press_event', moveon)
    plt.show()


