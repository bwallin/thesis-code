'''
Script for visualizing results of gibbs sampler.

Author: Bruce Wallin
'''
import pdb
import pickle
import sys
from optparse import OptionParser

from pylab import *

from misc import load_as_frame
from vis_lib import plot_data, plot_type_estimates, plot_ground_estimates, \
                    plot_canopy_estimates, plot_mcmc_diagnostics, plot_posterior_hist, \
                    compute_confusion_T, set_window_to_data

def main():
    usage = "usage: %prog [options] datafile"
    cmdline_parser = OptionParser(usage=usage)
    cmdline_parser.add_option('-s', '--start', dest='start',
                              type='int', default=None,
                              help='skip initial shots in track')
    cmdline_parser.add_option('-e', '--end', dest='end',
                              type='int', default=None,
                              help='end bound of shots in track')
    cmdline_parser.add_option('-m', '--meta-file', dest='meta_filename',
                              help='Meta data on data (i.e. simulation values and parameters).')
    options, args = cmdline_parser.parse_args()
    options, args = cmdline_parser.parse_args()

    results = pickle.load(open(args[0], 'rb'))
    validation_data = None
    if options.meta_filename:
        validation_data = pickle.load(open(options.meta_filename, 'rb'))

    # Load stuff into local namespace for convenience
    burnin = results['options'].burnin
    subsample = results['options'].subsample
    gibbs_iters = results['options'].iterations
    filename = results['filename']
    variables = results['variable_names']

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

    #phi_samples = array(results['gibbs_results']['phi']['samples'])
    diagnostic = results['gibbs_results']['diagnostic']

    print "burnin: %s" % burnin
    print "subsample: %s" % subsample
    print "iterations: %s" % gibbs_iters

    ### Load original data
    if 'data' in results:
        data = results['data']
    else:
        data = load_as_frame(filename, start=options.start, end=options.end)

    d_shot = sorted(list(set(data['d'])))
    shot_id = data['shot_id']
    signal_flag = data['signal_flag']
    d = data['d']
    z = data['z']
    N = len(z) # Number of data points

    if validation_data is not None:
        validation_data['p'], validation_data['q'] = validation_data['proportions'][:2]
        print 'g mean absolute error: %s' % (sum(abs(g_mean-validation_data['g'][:n]))/n)
        print 'g rms error: %s' % sqrt(sum((g_mean-validation_data['g'][:n])**2)/n)
        if 'T' in variables:
            confusion = compute_confusion_T(array(T_mode), signal_flag)
            print 'T confusion matrix: \n%s' % str(confusion)

    fig = figure()
    ax = fig.add_subplot(111)
    plot_data(ax, data, all_black=False)
    if validation_data is not None:
        plot(d, validation_data['g'][:n], 'b-')
        if 'h' in variables:
            plot(d, validation_data['h'][:n] + validation_data['g'][:n], 'g-')
    set_window_to_data(ax, data)

    fig = figure()
    ax = fig.add_subplot(111)
    if 'T' in variables:
        plot_type_estimates(ax, d, z, T_mode)
    ax.autoscale(False)
    plot_ground_estimates(ax, d_shot, g_mean, g_var)
    if 'h' in variables:
        plot_canopy_estimates(ax, d_shot, g_mean, g_var, h_mean, h_var)
    set_window_to_data(ax, data)


    plot_mcmc_diagnostics(diagnostic)

    fig = figure()
    histable = set(['phi', 'sigma_g', 'sigma_h', 'p', 'q'])
    for i, var in enumerate(histable.intersection(set(variables))):
        ax = fig.add_subplot(2, 3, i)
        samples = array(results['gibbs_results'][var]['samples'])
        plot_posterior_hist(ax, var, samples, validation_data)
    if 'T' in variables:
        ax = fig.add_subplot(2, 3, i+1)
        plot_posterior_hist(ax, 'p', p_type_samples[:,0], validation_data)
        ax = fig.add_subplot(2, 3, i+2)
        plot_posterior_hist(ax, 'q', p_type_samples[:,1], validation_data)
    show()

### Plots
if __name__=='__main__':
    main()
