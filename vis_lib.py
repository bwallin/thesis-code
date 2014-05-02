import pdb

from pylab import *

def data_plot(ax, data, elim_noise=False, all_black=False, marker='.', **kwargs):
    lines = []
    #ax.set_aspect('equal')
    data_sig = data[data['signal_flag'] != 0]
    x_sig, z_sig = data_sig['d'], data_sig['z']

    data_noise = data[data['signal_flag'] == 0]
    if len(data_noise > 0):
        x_noise, z_noise = data_noise['d'], data_noise['z']

    lines.extend(ax.plot(x_sig, z_sig, 'k'+marker, label="signal", **kwargs))
    if not elim_noise and len(data_noise) > 0:
        lines.extend(ax.plot(x_noise, z_noise, 'r'+marker if not all_black else 'k.', label=None if all_black else "noise", **kwargs))

    return lines


def set_window_to_data(ax, data):
    ax.set_xlim([min(data['d']), max(data['d'])])
    ax.set_ylim([min(data['z']), max(data['z'])])


def plot_data(ax, data, all_black=True):
    '''
    Plot true data.
    '''
    lines = data_plot(ax, data, elim_noise=False, all_black=all_black, marker='.', markersize=3)
    return lines

def plot_ground_estimates(ax, d_shot, g_mean, g_var):
    '''
    Plot ground and canopy surface estimates.
    '''
    lines = []
    lines += ax.plot(d_shot, g_mean, 'k', label="mean ground")
    lines += ax.plot(d_shot, g_mean+1.96*sqrt(g_var), 'k', linewidth=.5)
    lines += ax.plot(d_shot, g_mean-1.96*sqrt(g_var), 'k', linewidth=.5)
    return lines

def plot_canopy_estimates(ax, d_shot, g_mean, g_var, h_mean, h_var):
    '''
    Plot ground and canopy surface estimates.
    '''
    lines = []
    lines += ax.plot(d_shot, g_mean + h_mean, 'g-', label="mean canopy")
    lines += ax.plot(d_shot, g_mean + h_mean+1.96*sqrt(h_var + g_var), 'g-', linewidth=.5)
    lines += ax.plot(d_shot, g_mean + h_mean-1.96*sqrt(h_var + g_var), 'g-', linewidth=.5)
    return lines


def plot_type_estimates(ax, d, z, T_mode):
    '''
    Plot type modes.
    '''
    lines = []
    N = len(z)
    colors = {0:'r', 1:'k', 2:'g'}
    for i in range(N):
        lines += ax.plot([d[i]], [z[i]], '.', color=colors[T_mode[i]], markersize=3)
    return lines

def plot_mcmc_diagnostics(fig, diagnostic, burnin, subsample):
    lines = []
    variable, trace = diagnostic['variable'], diagnostic['trace']
    trace = array(trace).flatten()
    ax = fig.add_subplot(121)
    lines = ax.plot(trace)
    axvline(x=burnin, color='r')
    ax.set_ylabel(variable)
    ax.set_title('Trace of %s'%variable)
    maxlags = 50
    def normlize(x):
        return (x-mean(x))/var(x)
    ax = fig.add_subplot(122)
    lags, acf, l1, l2 = acorr(normlize(trace[burnin:]),
                              maxlags=maxlags, detrend=normlize)
    axvline(x=subsample, color='b')
    ax.set_xlim([0, maxlags])
    ax.set_ylabel('lag')
    ax.set_ylabel('ACF')
    ax.set_title('Autocorrelation in %s'%variable)
    return lines

def plot_iteration(evidence):
    '''
    Plot gibbs iteration.
    '''
    d, z, g, h, T = [evidence[var] for var in ['d', 'z', 'g', 'h', 'T']]
    d_shots = sorted(list(set(d)))
    fig = figure()
    ax = fig.add_subplot(111)
    colors = {0:'r', 1:'k', 2:'g'}
    for i in xrange(len(z)):
        ax.plot([d[i]], [z[i]], '.',
                color=colors[T[i]])
    ax.plot(d_shots, g, 'k-')
    ax.plot(d_shots, g + h, 'g-')
    show()

def compute_confusion_T(T, T_true):
    confusion = zeros((3,3))
    for i in range(3):
        for j in range(3):
            confusion[i,j] = sum((T==i) & (T_true==j))

    return confusion

def plot_posterior_hist(ax, variable, samples, validation_data=None):
    histogram = ax.hist(samples, 20)
    if validation_data:
        axvline(x=validation_data[variable])
    ax.set_xlabel(variable)


