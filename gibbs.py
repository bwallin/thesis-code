import pdb
from collections import defaultdict
import logging
import time
import pickle

from progressbar import ProgressBar, Percentage, Bar, ETA
from scipy import zeros, sum, dot, transpose, diag, mean

from stats_util import online_meanvar

class Model():
    def __init__(self):
        self.variable_names = None
        self.known_params = None
        self.hyper_params = None
        self.priors = None
        self.data = None
        self.initials = None
        self.FCP_samplers = None
        self.sample_handlers = None
        self.diagnostic_variable = None

    def set_variable_names(self, v_list):
        self.variable_names = v_list

    def set_known_params(self, p_dict):
        self.known_params = p_dict

    def set_hyper_params(self, p_dict):
        self.hyper_params = p_dict

    def set_initials(self, i_dict):
        self.initials = i_dict

    def set_priors(self, p_dict):
        self.priors = p_dict

    def set_FCP_samplers(self, f_dict):
        self.FCP_samplers = f_dict

    def set_sample_handlers(self, f_dict):
        self.sample_handlers = f_dict

    def set_diagnostic_variable(self, v):
        self.diagnostic_variable = v

    def set_data(self, data):
        self.data = data


class GibbsStep(object):
    def sample(self, model, evidence):
        pass

    def __call__(self, model, evidence):
        return self.sample(model, evidence)


class GibbsSampler(object):
    '''
    This class implements a generic gibbs sampler.
    '''
    def __init__(self, model, iterations, burnin, subsample):
        self.model = model # Model object: contains variables, parameters, distributions
        self.sample_handlers = model.sample_handlers # Dictionary of handlers for samples generated
        self.iterations = iterations
        self.burnin = burnin
        self.subsample = subsample
        self.visualizer = None
        self.set_diagnostic_variable(model.diagnostic_variable)
        self.bypass_vis = False

    def add_sample_handler(self, variable, handler):
        self.sample_handlers[variable].append(handler)

    def add_visualizer(self, visualizer):
        self.visualizer = visualizer

    def set_diagnostic_variable(self, variable):
        self.diagnostic_variable = variable
        self.diagnostic_trace = []

    def multi_run(self):
        '''
        Performs run of gibbs sampler
        '''
        # Initialize dictionary of evidence
        evidence = {}
        evidence.update(self.model.known_params)
        evidence.update(self.model.hyper_params)
        evidence.update(self.model.data)

        # Initial sample from priors
        for variable in self.model.variable_names:
            if self.model.initials and variable in self.model.initials:
                evidence[variable] = self.model.initials[variable]
            else:
                evidence[variable] = self.model.priors[variable].rvs()

        # Let the sampling begin
        for i in xrange(self.iterations):
            logging.info('Iteration: %s' % i)
            for variable in self.model.variable_names:
                logging.info('Sampling %s from FCP'%variable)
                FCP_draw_sample = self.model.FCP_samplers[variable]
                sample = FCP_draw_sample(self.model, evidence)
                evidence[variable] = sample
                if i > self.burnin and i % self.subsample == 0:
                    # Pass sample to its handlers
                    for handler in self.sample_handlers[variable]:
                        handler.update(sample)

            if self.diagnostic_variable:
                self.diagnostic_trace.append(evidence[self.diagnostic_variable])

        return

    def single_run(self):
        '''
        Performs run of gibbs sampler
        '''
        # Initialize dictionary of evidence
        evidence = {}
        evidence.update(self.model.known_params)
        evidence.update(self.model.hyper_params)
        evidence.update(self.model.data)

        # Initial sample from priors
        for variable in self.model.variable_names:
            if self.model.initials and variable in self.model.initials:
                evidence[variable] = self.model.initials[variable]
                logging.debug('Using initial for %s' % variable)
            else:
                evidence[variable] = self.model.priors[variable].rvs()
                logging.debug('Sampling %s from priors' % variable)

        if self.visualizer and not self.bypass_vis:
            self.visualizer(self, evidence)

        # Setup progressbar
        progressbar_widgets = ['Gibbs sampling:', Percentage(), ' ',
                               Bar('*'), ETA()]
        progress_bar = ProgressBar(widgets=progressbar_widgets, maxval=self.iterations).start()

        # Let the sampling begin
        for i in xrange(self.iterations):
            for variable in self.model.variable_names:
                logging.debug('Sampling %s from FCP'%variable)
                FCP_draw_sample = self.model.FCP_samplers[variable]
                sample = FCP_draw_sample(self.model, evidence)
                evidence[variable] = sample
                if i > self.burnin and i % self.subsample == 0:
                    # Pass sample to its handlers
                    for handler in self.sample_handlers[variable]:
                        handler.update(sample)

            if self.visualizer and i >= self.burnin and not self.bypass_vis:
                self.visualizer(self, evidence)

            if self.diagnostic_variable:
                self.diagnostic_trace.append(evidence[self.diagnostic_variable])

            progress_bar.update(i)
        progress_bar.finish()

    def results(self):
        '''
        Return a dictionary of results from sample handlers by variable.
        '''
        results = defaultdict(dict)
        for var, handlers in self.sample_handlers.iteritems():
            for handler in handlers:
                results[var].update(handler.result())
        if self.diagnostic_variable:
            results['diagnostic'] = {'variable':self.diagnostic_variable,
                                     'trace': self.diagnostic_trace}
        return results


def gibbs_worker(args):
    data, data_filename, outfile, iterations, burnin, subsample, model, params = args
    logging.info('Loading param file: %s'%params)
    params_module = __import__(params)
    logging.info('Done loading param file: %s'%params)
    logging.info('Loading model file: %s'%model)
    model_module = __import__(model)
    logging.info('Done loading model file: %s'%model)
    n = len(list(set(data.shot_id)))
    N = len(list(set(data)))
    model = model_module.define_model(params_module, data)

    sampler = GibbsSampler(
            model=model,
            iterations=iterations,
            burnin=burnin,
            subsample=subsample)

    start_time = time.time()
    logging.info('Starting sampler: %s' % outfile)
    sampler.multi_run()
    logging.info('Done sampler: %s' % outfile)
    processing_time = time.time() - start_time

    results = {}
    results['slice'] = slice(min(data.index), max(data.index))
    results['iterations'] = sampler.iterations
    results['burnin'] = sampler.burnin
    results['subsample'] = sampler.subsample 
    results['variable_names'] = model.variable_names
    results['known_params'] = model.known_params
    results['hyper_params'] = model.hyper_params
    results['data_filename'] = data_filename
    results['processing_time'] = processing_time
    results['gibbs_results'] = sampler.results()
    pickle.dump(results, open(outfile, 'wb'))


class generic_sample_handler():
    '''
    Template for sample handler.
    '''
    def __init__(self):
        pass

    def update(self, value):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError


class raw_sample_handler():
    def __init__(self):
        self.samples = []

    def update(self, value):
        self.samples.append(value)

    def result(self):
        return {'samples': self.samples}

class indep_meanvar_handler(generic_sample_handler):
    '''
    Computes mean and diag(cov) of multivariate.
    '''
    def __init__(self):
        self.online_meanvar = online_meanvar()

    def update(self, value):
        self.online_meanvar.update(value)

    def result(self):
        return {'mean': self.online_meanvar.mean, 'variance': self.online_meanvar.var}


class discrete_handler(generic_sample_handler):
    '''
    Computes array of pmfs for array of independent categorical variables.
    '''
    def __init__(self, support, length):
        self.counts = zeros((length, len(support)))
        self.length = length

    def update(self, x):
        self.counts[range(self.length), x.astype('int')] += 1

    def result(self):
        totals = sum(self.counts, axis=1)
        return {'pmf': [1./totals[i]*self.counts[i,:] for i in xrange(self.length)]}


