import pdb
from collections import defaultdict

from progressbar import ProgressBar, Percentage, Bar, ETA
from scipy import zeros, sum, dot, transpose, diag, mean

from stats_util import online_meanvar

class Model():
    def __init__(self):
        self.variable_names = None
        self.known_params = None
        self.hyper_params = None
        self.data = None
        self.priors = None
        self.initials = None
        self.FCP_samplers = None
        self.sample_handlers = None

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

    def set_data(self, d_dict):
        self.data = d_dict


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
        self.sample_handlers = defaultdict(list) # Dictionary of handlers for samples generated
        self.iterations = iterations
        self.burnin = burnin
        self.subsample = subsample
        self.visualizer = None
        self.diagnostic_variable = None
        self.bypass = False

    def add_sample_handler(self, variable, handler):
        self.sample_handlers[variable].append(handler)

    def add_visualizer(self, visualizer):
        self.visualizer = visualizer

    def set_diagnostic_variable(self, variable):
        self.diagnostic_variable = variable
        self.diagnostic_trace = []

    def run(self):
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

        # Setup progressbar
        progressbar_widgets = ['Gibbs sampling:', Percentage(), ' ',
                               Bar('*'), ETA()]
        progress_bar = ProgressBar(widgets=progressbar_widgets, maxval=self.iterations).start()

        # Let the sampling begin
        for i in xrange(self.iterations):
            for variable, FCP_sample in self.model.FCP_samplers.iteritems():
                sample = FCP_sample(self.model, evidence)
                evidence[variable] = sample
                if i > self.burnin and i % self.subsample == 0:
                    # Pass sample to its handlers
                    for handler in self.sample_handlers[variable]:
                        handler.update(sample)

            if self.visualizer and i > self.burnin and not self.bypass:
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
        return {'pmf': dot(diag(1./sum(self.counts, axis=1)),
                           self.counts)}


