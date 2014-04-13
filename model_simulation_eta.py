'''
Noise, ground, canopy, cover, AND multiple observations per shot.
'''
from __future__ import division
import sys
import pdb
import pickle

from pylab import array, zeros, mean, ones, eye, sqrt
from scipy import stats, ma, exp, log, nan, isnan, inf, isinf, logical_or
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
validation_filename = '../data/sims/simulation_zeta_3.pkl'
def define_model(data):
    # Builds model object
    if validation_filename is not None:
        # Known values from simulation (can use to initialize sampler/skip burn-in)
        p = pickle.load(open(validation_filename, 'rb'))
        g = p['g']
        h = p['h']
        T = p['T']
        C = p['C']
        noise_proportion = p['noise_proportion']
        transition_var_g = p['sigma_g']**2
        transition_var_h = p['sigma_h']**2
        # Also needed for known parameters (so load the right pkl dummy!)
        phi = p['phi']
        mu_h = p['mu_h']
        observation_var_g = p['sigma_z_g']**2
        observation_var_h = p['sigma_z_h']**2
        canopy_cover = p['canopy_cover']
        cover_transition_matrix = p['cover_transition_matrix']
        z_min, z_max = -60, 60
    else:
        phi = None
        mu_h = None
        observation_var_g = None
        observation_var_h = None
        canopy_cover = array([.0, .5, 1.]) 
        cover_transition_matrix = array([[.99, .005, .005], [.005, .99, .005], [.005, .005, .99]])
        z_min, z_max = -60, 60

    z = data.get('z')
    shot_id = data.get('shot_id')
    N = len(z) # Number of data points
    n = len(set(shot_id)) # Number of shots

    # Variables to be sampled
    variable_names = ['g', 'h', 'T', 'C', 'noise_proportion', 'transition_var_g', 'transition_var_h']

    known_params = {'observation_var_g': observation_var_g, # observation stdev
                    'observation_var_h': observation_var_h,
                    'mu_h': mu_h, # mean canopy
                    'phi': phi, # canopy autoreg param
                    'z_min': z_min,
                    'z_max': z_max,
                    'canopy_cover': canopy_cover, # possible canopy cover states
                    'cover_transition_matrix': cover_transition_matrix} # canopy transition matrix
    hyper_params = {'g': {'mu': zeros(n), 'cov': 1000*eye(n)}, # mvn prior
                    'h': {'mu': zeros(n), 'cov': 1000*eye(n)}, # mvn prior
                    'T': {'p': array([1/3., 1/3., 1/3.])}, # iid categorical prior
                    'C': {'p': array([1/3., 1/3., 1/3.])}, # iid categorical prior
                    'noise_proportion':{'alpha': array((1.,1.))}, # dirichlet prior
                    'transition_var_g': {'a': 11, 'b': .1, 'max': 1}, # inv-gamma prior
                    'transition_var_h': {'a': 11, 'b': 40, 'max': 9}} # inv-gamma prior
    initials = {'g': -25+zeros(n),
                'h': 30*ones(n),
                'T': array([(0 if abs(z[i]+25)>1 else 1) for i in xrange(n)]),
                'C': zeros(n, dtype=int),
                'noise_proportion': .2,
                'transition_var_g': .1,
                'transition_var_h': 1}
    initials = {'g': g[:N],
                'h': h[:N],
                'C': C[:N],
                'T': T[:N],
                'noise_proportion': noise_proportion,
                'transition_var_g': transition_var_g,
                'transition_var_h': transition_var_h}
    priors = {'g': mvnorm(hyper_params['g']['mu'], hyper_params['g']['cov']),
              'h': mvnorm(hyper_params['h']['mu'], hyper_params['h']['cov']),
              'C': iid_dist(categorical(hyper_params['C']['p']), N),
              'T': iid_dist(categorical(hyper_params['T']['p']), N),
              'noise_proportion': dirichlet(hyper_params['noise_proportion']['alpha']),
              'transition_var_g': stats.invgamma(hyper_params['transition_var_g']['a'], scale=hyper_params['transition_var_g']['b']),
              'transition_var_h': stats.invgamma(hyper_params['transition_var_h']['a'], scale=hyper_params['transition_var_h']['b'])}
    FCP_samplers = {'g': ground_elev_step(),
                    'h': canopy_height_step(),
                    'C': cover_step(),
                    'T': type_step(),
                    'noise_proportion': noise_proportion_step(),
                    'transition_var_g': transition_var_g_step(),
                    'transition_var_h': transition_var_h_step()}

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
class ground_elev_step(GibbsStep):
    def __init__(self, *args, **kwargs):
        super(ground_elev_step, self).__init__(*args, **kwargs)
        self._kalman = KalmanFilter()

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

        z_g = ma.asarray(z.copy().reshape((N, 1)))
        z_g[T == 0] = nan # mask noise points
        z_g[T == 0] = ma.masked
        z_g[T == 2] -= h[T == 2] # get ground measurement from canopy top

        obs_cov = ma.asarray(inf + zeros((n,1,1)))
        obs_cov[T == 1] = observation_var_g
        obs_cov[T == 2] = observation_var_h
        obs_cov[isinf(obs_cov)] = ma.masked

        kalman = self._kalman
        kalman.initial_state_mean = array([prior_mu_g[0],])
        kalman.initial_state_covariance = array([prior_cov_g[0,0],])
        kalman.transition_matrices = eye(1)
        kalman.transition_covariance = array([transition_var_g,])
        kalman.observation_matrices = eye(1)
        kalman.observation_covariance = obs_cov
        sampled_g = forward_filter_backward_sample(kalman, z_g)

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

        z_h = ma.asarray(z.copy().reshape((N, 1)))
        z_h[logical_or(T == 0, T == 1)] = nan # mask noise and ground values
        z_h[isnan(z_h)] = ma.masked # how missing values are formatted in pykalman

        obs_cov = ma.asarray(inf + zeros((n, 1, 1)))
        obs_cov[T == 2] = observation_var_h
        obs_cov[isinf(obs_cov)] = ma.masked

        kalman = self._kalman
        kalman.initial_state_mean = array([prior_mu_h[0],])
        kalman.initial_state_covariance = array([prior_cov_h[0,0],])
        kalman.transition_matrices = array([phi,])
        kalman.transition_covariance = array([transition_var_h,])
        kalman.transition_offsets = mu_h*(1-phi)*ones((n, 1))
        kalman.observation_matrices = eye(1)
        kalman.observation_covariance = obs_cov
        sampled_h = forward_filter_backward_sample(kalman, z_h)

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

        n = len(h)

        h_var_posterior = stats.invgamma(a + (n-1)/2., scale=b + sum((h[1:] - h[:-1])**2)/2.)
        h_var = h_var_posterior.rvs()
        return min(h_var, max_var)


class type_step(GibbsStep):
    def sample(self, model, evidence):
        g = evidence['g']
        h = evidence['h']
        noise_proportion = evidence['noise_proportion']
        z = evidence['z']
        observation_var_g = evidence['observation_var_g']
        observation_var_h = evidence['observation_var_h']
        C = evidence['C']
        shot_id = evidence['shot_id']

        canopy_cover = model.known_params['canopy_cover']
        z_min = model.known_params['z_min']
        z_max = model.known_params['z_max']

        N = len(z)
        T = zeros(N)
        noise_rv = stats.uniform(z_min, z_max - z_min)
        for i in xrange(N):
            l = zeros(3)
            l[0] = noise_proportion*noise_rv.pdf(z[i])
            g_norm = stats.norm(g[shot_id[i]], observation_var_g)
            l[1] = (1-noise_proportion)*(1-canopy_cover[C[shot_id[i]]])*g_norm.pdf(z[i])
            h_norm = stats.norm(h[shot_id[i]], observation_var_h)
            l[2] = (1-noise_proportion)*(canopy_cover[C[shot_id[i]]])*h_norm.pdf(z[i] - g[shot_id[i]])
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
        alpha = model.hyper_params['noise_proportion']['alpha']
        T = evidence['T']
        n = len(T)
        n_noise = sum(T==0)
        return dirichlet(alpha + (n_noise, n - n_noise)).rvs()[0]


def visualize_gibbs(sampler, evidence):
    pdb.set_trace()
    z, T, C, d, g, h, transition_var_g, transition_var_h, canopy_cover = [evidence[var] for var in ['z', 'T', 'C', 'd', 'g', 'h', 'transition_var_g', 'transition_var_h', 'canopy_cover']]
    g = g.reshape((len(g), ))
    h = h.reshape((len(h), ))

    print "transition_var_g: %s" % transition_var_g
    print "transition_var_h: %s" % transition_var_h
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


