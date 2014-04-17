from __future__ import division
import pdb
import os
import sys

from scipy import array, zeros, dot, transpose, matrix
from scipy import stats, sqrt, cumsum
from scipy import loadtxt
from scipy.linalg import norm, inv
from stats_util import MVNormal
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
    frame[['shot_id', 'index', 'signal_flag']] = frame[['shot_id', 'index', 'signal_flag']].astype(int)
    frame.filepath = filepath

    return frame


def forward_filter_backward_sample(kf, y):
    '''
    Sample from the DLM kf conditioned on observations y.
    '''
    dim_obs = y.shape
    n_obs = y.shape[0]
    n_states = len(kf.initial_state_mean)
    G, W = kf.transition_matrices, kf.transition_covariance
    u = kf.transition_offsets
    if u is None:
        u = zeros((n_obs, n_states))

    # filtered estimate
    x_f, P_f = kf.filter(y)

    # one-step predicted estimate
    x_p, P_p = zeros((n_obs, n_states)), zeros((n_obs, n_states, n_states))
    x_p[0,:] = kf.initial_state_mean
    P_p[0,:,:] = kf.initial_state_covariance
    for i in xrange(1, n_obs):
        x_p[i] = dot(G, x_f[i-1]) + u[i,:]
        P_p[i] = dot(G, dot(P_f[i-1], transpose(G))) + W

    # Sample last state, and iterate backwards
    x = zeros((n_obs, n_states))
    x[-1] = MVNormal(x_f[-1], P_f[-1]).rvs()
    for i in range(n_obs-2, -1, -1):
        J = dot(P_f[i], dot(transpose(G), inv(matrix(P_p[i+1]))))
        m = x_f[i] + dot(J, x[i+1] - x_p[i+1])
        V = P_f[i] - dot(J, dot(P_p[i+1], transpose(J)))

        N = MVNormal(m, V)
        x[i] = N.rvs()

    return x

