import pdb

from pylab import *
from matplotlib import pyplot

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


def plot_data(ax, data, all_black=True, markersize=3, elim_noise=False):
    '''
    Plot true data.
    '''
    lines = data_plot(ax, data, elim_noise=elim_noise, all_black=all_black, marker='.', markersize=markersize)
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

def plot_canopy_estimates(ax, d_shot, g_mean, g_var, h_mean, h_var, mask=None):
    '''
    Plot ground and canopy surface estimates.
    '''
    lines = []
    h_mean = ma.asarray(h_mean)
    h_mean[mask] = ma.masked
    lines += ax.plot(d_shot, g_mean + h_mean, 'k--', label="mean canopy")
    lines += ax.plot(d_shot, g_mean + h_mean+1.96*sqrt(h_var + g_var), 'k--', linewidth=.5)
    lines += ax.plot(d_shot, g_mean + h_mean-1.96*sqrt(h_var + g_var), 'k--', linewidth=.5)
    return lines


def plot_type_estimates(ax, d, z, T_mode, markersize=5, alpha=1):
    '''
    Plot type modes.
    '''
    labels = ['noise', 'ground', 'canopy']
    labeled = [False, False, False]
    N = len(z)
    colors = {0:'r', 1:'k', 2:'g'}
    lines = []
    for i in range(N):
        if not labeled[T_mode[i]]: 
            label = labels[T_mode[i]]
            labeled[T_mode[i]] = True
        else:
            label = None
        lines += ax.plot([d[i]], [z[i]], '.', color=colors[T_mode[i]], markersize=markersize, alpha=alpha, label=label)
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
    ylim = ax.get_ylim()
    ax.text(burnin, mean(ylim), 'burn-in', rotation='vertical', ha='right', color='red')

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
    ylim = ax.get_ylim()
    ax.text(subsample, mean(ylim), 'subsample', rotation='vertical', ha='right', color='blue')
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
    pyplot.locator_params(axis = 'x', nbins = 4)
    histogram = ax.hist(samples, bins=12)
    if validation_data:
        axvline(x=validation_data[variable])
    ax.set_xlabel(variable)
    ax.set_ylim(0, max(histogram[0])*1.1)
    #xlim = ax.get_xlim()
    #xlimw = xlim[1]-xlim[0]
    #ax.set_xlim(xlim[0]-.1*xlimw, xlim[1]+.1*xlimw)
