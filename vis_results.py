'''
Script for visualizing results of gibbs sampler.

Author: Bruce Wallin

Usage:
    python vis_results.py path/to/results.pkl
'''
import pdb
import pickle
import sys
from optparse import OptionParser

from pylab import *
from matplotlib import pyplot as plt
from misc import load_as_frame
from vis_lib import plot_data, plot_type_estimates, plot_ground_estimates, \
                    plot_canopy_estimates, plot_mcmc_diagnostics, plot_posterior_hist, \
                    compute_confusion_T, set_window_to_data

def main():
    usage = "usage: %prog [options] datafile"
    cmdline_parser = OptionParser(usage=usage)
    cmdline_parser.add_option('-v', '--validation-data', dest='validation_filename',
                              help='Validation data (i.e. simulation values and parameters).')
    cmdline_parser.add_option('-o', '--output-name', dest='output_name',
                              help='Save series of plots under this name.')
    options, args = cmdline_parser.parse_args()
    options, args = cmdline_parser.parse_args()

    results = pickle.load(open(args[0], 'rb'))
    validation_data = None
    if options.validation_filename:
        validation_data = pickle.load(open(options.validation_filename, 'rb'))

    # Load results for convenience
    start = results['options'].start
    end = results['options'].end
    burnin = results['options'].burnin
    subsample = results['options'].subsample
    gibbs_iters = results['options'].iterations
    filename = results['filename']
    variables = results['variable_names']
    diagnostic = results['gibbs_results']['diagnostic']

    if 'T' in variables:
        T_pmf = results['gibbs_results']['T']['pmf']
        T_mode = [list(T_pmf[i]).index(max(T_pmf[i])) for i in xrange(len(T_pmf))]
    p_type_samples = array(results['gibbs_results']['p_type']['samples'])
    g_mean = results['gibbs_results']['g']['mean']
    n = len(g_mean)
    g_mean = g_mean.reshape((n,))
    g_var = results['gibbs_results']['g']['variance']
    g_var = g_var.reshape((n,))
    sigma_g_samples = array(results['gibbs_results']['sigma_g']['samples'])
    if 'h' in variables:
        h_mean = results['gibbs_results']['h']['mean']
        h_mean = h_mean.reshape((n,))
        h_var = results['gibbs_results']['h']['variance']
        h_var = h_var.reshape((n,))
        sigma_h_samples = array(results['gibbs_results']['sigma_h']['samples'])
    if 'phi' in variables:
        phi_samples = array(results['gibbs_results']['phi']['samples'])


    ### Load original data
    data = load_as_frame(filename, start=start, end=end)

    d_shot = sorted(list(set(data['d'])))
    shot_id = data['shot_id']
    signal_flag = data['signal_flag']
    d = data['d']
    z = data['z']
    N = len(z) # Number of data points

    # Print out information
    print "N: %s" % N
    print "burnin: %s" % burnin
    print "subsample: %s" % subsample
    print "iterations: %s" % gibbs_iters
    if validation_data is not None:
        validation_data['p'], validation_data['q'] = validation_data['proportions'][:2]
        print 'g mean absolute error: %s' % (sum(abs(g_mean-validation_data['g'][:n]))/n)
        print 'g rms error: %s' % sqrt(sum((g_mean-validation_data['g'][:n])**2)/n)
        if 'T' in variables:
            confusion = compute_confusion_T(array(T_mode), signal_flag)
            print 'T confusion matrix: rows*columns = T_mode*T_True\n%s' % str(confusion)
    if 'h' in variables:
        print 'h mean absolute error: %s' % (sum(abs(h_mean-validation_data['h'][:n]))/n)
        print 'h rms error: %s' % sqrt(sum((h_mean-validation_data['h'][:n])**2)/n)


    # Draw plots

    fig_profile = figure()
    mng = get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    ax = fig_profile.add_subplot(311)
    plot_data(ax, data, all_black=True)
    ax = fig_profile.add_subplot(312, sharex=ax, sharey=ax)
    plot_data(ax, data, all_black=False)
    if validation_data is not None:
        plot(d, validation_data['g'][:n], 'b-', alpha=.5)
        if 'h' in variables:
            plot(d, validation_data['h'][:n] + validation_data['g'][:n], 'g-', alpha=.5)
    set_window_to_data(ax, data)
    ax.set_title('Data (and validation)')

    ax = fig_profile.add_subplot(313, sharex=ax, sharey=ax)
    if 'T' in variables:
        plot_type_estimates(ax, d, z, T_mode)
    ax.autoscale(False)
    plot_ground_estimates(ax, d_shot, g_mean, g_var)
    if 'h' in variables:
        plot_canopy_estimates(ax, d_shot, g_mean, g_var, h_mean, h_var)
    set_window_to_data(ax, data)
    ax.set_title('Sampler results')


    fig_diag = figure()
    mng = get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plot_mcmc_diagnostics(fig_diag, diagnostic, burnin, subsample)

    fig_hist = figure()
    mng = get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    histable = set(['phi', 'sigma_g', 'sigma_h', 'p', 'q'])
    for i, var in enumerate(histable.intersection(set(variables))):
        ax = fig_hist.add_subplot(2, 3, i)
        samples = array(results['gibbs_results'][var]['samples'])
        plot_posterior_hist(ax, var, samples, validation_data)
    if 'T' in variables:
        ax = fig_hist.add_subplot(2, 3, i+1)
        plot_posterior_hist(ax, 'p', p_type_samples[:,0], validation_data)
        ax = fig_hist.add_subplot(2, 3, i+2)
        plot_posterior_hist(ax, 'q', p_type_samples[:,1], validation_data)

    if options.output_name:
        fig_profile.savefig('../figs/'+options.output_name+'_profile.png')
        fig_diag.savefig('../figs/'+options.output_name+'_diag.png')
        fig_hist.savefig('../figs/'+options.output_name+'_hist.png')
    else:
        show()

### Plots
if __name__=='__main__':
    main()
