'''
This script implements a Gibbs sampler for a canopy height model (defined in model.py)

Author: Bruce Wallin
'''
import pdb
import time
from optparse import OptionParser
import logging
import pickle

from scipy import array

# local imports
from gibbs import GibbsSampler
from misc import load_as_frame

def main():
    ### Parse command line options
    usage = "usage: %prog [options] datafile"
    cmdline_parser = OptionParser(usage=usage)
    cmdline_parser.add_option('-o', '--output', dest='output_filename', metavar='FILE',
                              default='results.pkl',
                              help='Serialize results to pickle FILE')
    cmdline_parser.add_option('-m', '--model', dest='model', metavar='FILE',
                              default='model',
                              help='Choose model to use')
    cmdline_parser.add_option('-v', '--verbose', dest='loglevel', default=logging.WARNING,
                              action='store_const', const=logging.DEBUG,
                              help='Debug logging level mode')
    cmdline_parser.add_option('-i', '--iterations', dest='iterations',
                              type='int', default=25000,
                              help='Number of Gibbs iterations')
    cmdline_parser.add_option('-b', '--burnin', dest='burnin',
                              type='int', default=500,
                              help='Number of burn-in iterations')
    cmdline_parser.add_option('-s', '--subsample', dest='subsample',
                              type='int', default=10,
                              help='Subsample rate')
    cmdline_parser.add_option('-t', '--start', dest='start',
                              type='int', default=None,
                              help='start')
    cmdline_parser.add_option('-e', '--end', dest='end',
                              type='int', default=None,
                              help='end')
    cmdline_parser.add_option('-g', '--visualize', dest='visualize',
                              action='store_true',
                              help='Visualize intermediate results')
    cmdline_parser.add_option('-G', '--visualize-priors', dest='visualize_priors',
                              action='store_true',
                              help='Visualize prior distributions')
    cmdline_parser.add_option('-p', '--parameter-file', dest='parameter_filename',
                              help='Use known parameters in file (i.e. simulated file).')
    options, args = cmdline_parser.parse_args()

    logging.basicConfig(level=options.loglevel, format='%(asctime)s %(message)s')
    data_filename = args[0]
    gibbs_iters = options.iterations
    burnin = options.burnin
    subsample = options.subsample
    model_module = __import__(options.model)

    if options.parameter_filename is not None:
        known_params = pickle.load(open(options.parameter_filename, 'rb'))

    # Build model and load data
    data = load_as_frame(data_filename, start=options.start, end=options.end)
    model = model_module.define_model(data)
    n_points = len(data)
    n_shots = len(set(data['shot_id']))

    # Setup gibbs sampler
    sampler = GibbsSampler(model=model,
                           iterations=gibbs_iters,
                           burnin=burnin,
                           subsample=subsample)
    if options.visualize:
        sampler.add_visualizer(model_module.visualize_gibbs)
    if options.visualize_priors:
        model_module.visualize_priors(model.priors)

    # Begin sampling
    start_time = time.time()
    sampler.run()
    gibbs_time = time.time() - start_time
    print "Gibbs sampler ran %d.1 minutes" % (gibbs_time/60.)

    # Write out results
    results = {}
    results['options'] = options
    results['variable_names'] = model.variable_names
    results['known_params'] = model.known_params
    results['hyper_params'] = model.hyper_params
    results['filename'] = data_filename
    results['data'] = data
    results['gibbs_results'] = sampler.results()
    pickle.dump(results, open(options.output_filename, 'wb'))


if __name__=='__main__':
    main()
