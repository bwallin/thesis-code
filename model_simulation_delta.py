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
    p = pickle.load(open('../data/sims/simulation_delta_0.pkl', 'rb'))
    g = p['g']
    h = p['h']
    T = p['T']
    sigma_g = p['sigma_g']
    sigma_h = p['sigma_h']
    # Also needed for known parameters (so load the right pkl dummy!)
    phi = p['phi']
    mu_h = p['mu_h']
    sigma_z_g = p['sigma_z_g']
    sigma_z_h = p['sigma_z_h']

    n = len(data)
    z = data.get('z')
    variable_names = ['g', 'h', 'T', 'p_type', 'sigma_g', 'sigma_h']
    known_params = {'sigma_z_g': sigma_z_g,
                    'sigma_z_h': sigma_z_h,
                    'mu_h': mu_h,
                    'phi': phi}
    hyper_params = {'alpha_type': array((1., 1., 1.)),
                    'prior_mu_g': -25.+zeros(n),
                    'prior_cov_g': 100.*eye(n),
                    'prior_mu_h': 30.+zeros(n),
                    'prior_cov_h': 100.*eye(n),
                    'a_g': 11,
                    'b_g': .1,
                    'a_h': 11,
                    'b_h': 40}
    #initials = {'g': -25+zeros(n),
    #            'sigma_g': .1,
    #            'sigma_h': 1,
    #            'T': array([(0 if abs(z[i]+25)>1 else 1) for i in xrange(n)])}
    initials = {'sigma_g': sigma_g,
                'sigma_h': sigma_h,
                'p_type': array((0, .5, .5)),
                'T': T[:n],
                'g': g[:n],
                'h': h[:n]}
    priors = {'p_type': dirichlet(hyper_params['alpha_type']),
              'sigma_g': stats.invgamma(hyper_params['a_g'], scale=hyper_params['b_g']),
              'sigma_h': stats.invgamma(hyper_params['a_h'], scale=hyper_params['b_h']),
              'g': mvnorm(hyper_params['prior_mu_g'], hyper_params['prior_cov_g']),
              'h': mvnorm(hyper_params['prior_mu_h'], hyper_params['prior_cov_h']),
              'T': iid_dist(categorical(hyper_params['alpha_type']), n)}
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
        return min(sqrt(h_var), 3) # keep sigma_h from possibly blowing up


class phi_step(GibbsStep):
    def __init__(self, *args, **kwargs):
        super(phi_step, self).__init__(*args, **kwargs)
        self.acceptance_rate = None
        self.n = 0

    def update_acceptance_rate(self, rate):
        if self.acceptance_rate:
            (self.acceptance_rate*self.n + rate)/(self.n+1)
        else:
            self.acceptance_rate = rate
        self.n += 1

    def sample(self, model, evidence):
        # Metropolis-Hastings (based on Kim, Shephard and Chib(1998))
        MAX_REJECT = 100000
        h, sigma_h = evidence['h'], evidence['sigma_h']
        current_phi = evidence['phi']
        alpha = model.hyper_params['alpha_phi']
        mu_h = model.known_params['mu_h']

        g = lambda phi: (alpha[0] - 1)*log((1 + phi)/2.) + (alpha[1] - 1)*log((1 - phi)/2.) - \
                        (h[0] - mu_h)**2*(1 - phi**2)/(2*sigma_h**2) + \
                        1/2.*log(1 - phi**2)

        for i in xrange(MAX_REJECT):
            proposal_mean = sum((h[1:] - mu_h)*(h[:-1] - mu_h))/sum((h[:-1] - mu_h)**2)
            proposal_var = sigma_h**2/sum((h[:-1] - mu_h)**2)
            try:
                proposed_phi = stats.norm(proposal_mean, sqrt(proposal_var)).rvs()
            except:
                pdb.set_trace()
            if -1 < proposed_phi < 1:
                accept_rv = stats.bernoulli(max(min(exp(g(proposed_phi) - g(current_phi)), 1), 0))
                if accept_rv.rvs():
                    self.update_acceptance_rate(1./(i + 1))
                    return proposed_phi

        raise Exception('Metropolis-Hastings rejected too many: %d'%MAX_REJECT)


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
            p = l/sum(l)
            T[i] = categorical(p).rvs()

        return T


class p_type_step(GibbsStep):
    def sample(self, model, evidence):
        alpha_type = model.hyper_params['alpha_type']
        T = evidence['T']
        m = len(alpha_type)
        return dirichlet(alpha_type + array([sum(T==i) for i in xrange(m)])).rvs()


def visualize_gibbs(sampler, evidence):
    z, T, d, g, h, sigma_g, sigma_h = [evidence[var] for var in ['z', 'T', 'd', 'g', 'h', 'sigma_g', 'sigma_h']]
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
    plt.plot(d, g+h, 'g-', linewidth=3, alpha=.5)
    #plt.fill_between(d, g-1.96*sigma_g, g+1.96*sigma_g, color='k', alpha=.3)
    def moveon(event):
        plt.close()
    fig.canvas.mpl_connect('key_press_event', moveon)
    plt.show()


