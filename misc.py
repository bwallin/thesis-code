from __future__ import division
import pdb
import os
import sys

from scipy import array, zeros, dot, transpose, matrix
from scipy import stats, sqrt, cumsum
from scipy import loadtxt
from scipy.linalg import norm, inv
import pandas

def gen_shot_ids(frame, tol=.05):
    """Groups individual ranges into shots and assigns ID

    Returns shot id array"""
    sids = zeros(len(frame))
    sids_dict = {}
    sid = 0
    for i in range(len(frame)):
        di = frame.d[i]
        recorded = False
        for k,d in sids_dict.iteritems():
            if abs(di - d) < tol:
                sids[i] = k
                recorded = True

        if not recorded:
            sids[i] = sid
            sids_dict[sid] = di
            sid += 1

    return sids

def load_as_frame(filepath, start=None, end=None):
    '''
    Read data from Sigma icesat-II simulations: http://icesat.gsfc.nasa.gov/icesat2/data/sigma/sigma_data.php

    returns pandas dataframe
    '''
    raw = loadtxt(open(filepath, 'r'))
    raw = raw[start:end]
    columns = ['x', 'y', 'z', 'index', 'signal_flag']
    data_dict = {}
    for col_ind, col in enumerate(columns):
        data_dict[col] = raw[:, col_ind]

    frame = pandas.DataFrame(data_dict)

    # 2D model uses along track distance d
    xy = array(zip(frame['x'], frame['y']))
    d = zeros(len(frame))
    d[1:] = cumsum([sqrt(norm(xy[i] - xy[i-1])) for i in xrange(1, len(frame))])
    frame['d'] = d

    # Assign id to points detected from same pulse
    frame['shot_id'] = gen_shot_ids(frame)

    return frame


def forward_filter_backward_sample(kf, y):
    '''
    Sample from the DLM kf conditioned on observations y.
    '''
    n = len(y)
    m = kf.n_dim_state
    G, W = kf.transition_matrices, kf.transition_covariance

    # filtered estimate
    x_f, P_f = kf.filter(y)

    # one-step predicted estimate
    x_p, P_p = zeros((n,m)),zeros((n,m,m))
    x_p[0] = kf.initial_state_mean
    P_p[0] = kf.initial_state_covariance
    for i in xrange(1,n):
        x_p[i] = G*x_f[i-1]
        P_p[i] = dot(G, dot(P_f[i-1], transpose(G))) + W

    # Sample last state, and iterate backwards
    x = zeros((n,m))
    x[-1] = stats.norm(loc=x_f[-1], scale=sqrt(P_f[-1])).rvs()
    u = kf.transition_offsets if kf.transition_offsets is not None else zeros((n,m))
    for i in range(n-2, -1, -1):
        J = dot(P_f[i], dot(transpose(G), inv(matrix(P_p[i+1]))))
        m = x_f[i] + dot(J, x[i+1] - u[i+1] - x_p[i+1])
        V = P_f[i] - dot(J, dot(P_p[i+1], transpose(J)))

        N = stats.norm(m, sqrt(V))
        x[i] = N.rvs()

    return x

