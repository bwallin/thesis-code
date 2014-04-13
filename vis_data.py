'''
Script for visualizing sigma MPL profile data.

Author: Bruce Wallin
'''
import pdb
import pickle
import sys
from optparse import OptionParser

from pylab import *

from misc import load_as_frame
from vis_lib import plot_data, plot_type_estimates, plot_ground_estimates, \
                    plot_canopy_estimates, plot_mcmc_diagnostics

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
    cmdline_parser.add_option('-o', '--output-to-file', dest='output',
                              action='store_true',
                              help='Write to image file <filename>.png')
    cmdline_parser.add_option('-b', '--all-black', dest='all_black',
                              action='store_true',
                              help='All points black')

    options, args = cmdline_parser.parse_args()

    filenames = args
    for filename in filenames:
        data = load_as_frame(filename, start=options.start, end=options.end)
        d = data['d']
        validation_data = None
        if options.meta_filename:
            validation_data = pickle.load(open(options.meta_filename, 'r'))

        fig = figure(figsize=(16,6))
        ax = fig.add_subplot(111)
        plot_data(ax, data, all_black=options.all_black)
        if validation_data is not None:
            plot(d, validation_data['g'], 'k-')

        if options.output:
            savefig(filename+'.png')
        else:
            show()

if __name__ == '__main__':
    main()
