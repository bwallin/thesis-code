from __future__ import division
from collections import defaultdict
import operator
import math

from numpy import cumsum, zeros, dot
from numpy import exp, power, log, pi
from numpy.linalg import det, inv
from numpy.random import dirichlet as dirichlet_rvs
from numpy.random import multivariate_normal, rand

class dirichlet():
    '''
    Dirichlet random variable.
    '''
    def __init__(self, alpha):
        self.alpha = alpha

    def rvs(self):
        return dirichlet_rvs(self.alpha)

    def pdf(self, x):
        # UNTESTED!
        alpha = self.alpha
        return (math.gamma(sum(alpha)) /
               reduce(operator.mul, [math.gamma(a) for a in alpha]) *
               reduce(operator.mul, [x[i]**(alpha[i]-1.0) for i in range(len(alpha))]))


class categorical():
    '''
    Categorical random variable.
    '''
    def __init__(self, probs):
        self.pmf = probs

    def rvs(self, size=1):
        cdf = cumsum(self.pmf)
        cdf /= max(cdf)
        rvs = zeros((size,), dtype='Int32')
        for i in xrange(size):
            rvs[i] = cdf.searchsorted(rand())

        return rvs


class mvnorm():
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def rvs(self, size=None):
        return multivariate_normal(self.mean, self.cov, size)

    def pdf(self, x):
        # UNTESTED
        mean, cov = self.mean, self.cov
        k = x.shape[0]
        part1 = exp(-0.5*k*log(2*pi))
        part2 = power(det(cov),-0.5)
        dev = x-mean
        part3 = exp(-0.5*dot(dot(dev.transpose(), inv(cov)), dev))
        return part1*part2*part3


class iid_dist():
    def __init__(self, distribution, n):
        self.distribution = distribution
        self.n = n

    def rvs(self, size=None):
        return self.distribution.rvs(self.n)


class online_meanvar():
    def __init__(self):
        self.mean = None
        self.var = None
        self.n_samples = 0

    def update(self, value):
        n_samples, mean, var = self.n_samples, self.mean, self.var
        n_samples += 1
        if mean is None:
            mean = 0
            var = 0
        delta = value - mean
        mean += delta/n_samples
        var = (var*(n_samples-1) + delta*(value - mean))/n_samples
        self.n_samples, self.mean, self.var = n_samples, mean, var

    def result(self):
        return (self.mean, self.var)

class online_mode():
    def __init__(self):
        self.n_samples = 0
        self.counts = defaultdict(int)
        self.mode = None

    def update(self, value):
        self.counts[value] += 1
        self.n_samples += 1
        self.mode = max(self.counts, key=self.counts.get)

    def result(self):
        return self.mode
