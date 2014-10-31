from __future__ import division
import pdb
import os
import sys

from scipy import array, zeros, dot, transpose, matrix
from scipy import stats, sqrt, cumsum, arange, logical_and
from scipy import loadtxt, ones
from scipy.linalg import norm, inv
from stats_util import MVNormal
import pandas
import h5py

from projutil import make_utm_transform

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
    if 'matlas' in filepath:
        return _load_matlas_as_frame(filepath, start=start, end=end)
    elif 'cp.csv.gz' in filepath:
        return _load_gliht_as_frame(filepath, start=start, end=end)
    else:
        return _load_sigma_as_frame(filepath, start=start, end=end)


def _load_sigma_as_frame(filepath, start=None, end=None):
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


def _load_matlas_as_frame(filepath, start=None, end=None):
    '''
    Read data from MATLAS icesat-II simulations: http://icesat.gsfc.nasa.gov/icesat2/data/matlas

    returns pandas dataframe
    '''
    f = h5py.File(filepath, 'r')
    lat = f['photon']['channel_strong']['latitude'][start:end]
    lon = f['photon']['channel_strong']['longitude'][start:end]
    elev = f['photon']['channel_strong']['elev'][start:end]
    shot_num = f['photon']['channel_strong']['shot_num'][start:end]
    x,y = make_utm_transform(17)(lon,lat)

    mask = logical_and(50<array(elev), array(elev)<175)
    lat = lat[mask]
    lon = lon[mask]
    elev = elev[mask]
    x = x[mask]
    y = y[mask]
    shot_num = shot_num[mask]

    columns = ['x', 'y', 'z', 'index', 'signal_flag']
    data_dict = {}
    data_dict['x'] = x
    data_dict['y'] = y
    data_dict['z'] = elev

    data_dict['index'] = arange(len(data_dict['z']))
    data_dict['signal_flag'] = array(shot_num)!=-999

    frame = pandas.DataFrame(data_dict)

    # 2D model uses along track distance d
    xy = array(zip(x, y))
    d = zeros(len(frame))
    d[1:] = cumsum([sqrt(norm(xy[i] - xy[i-1])) for i in xrange(1, len(frame))])
    frame['d'] = d

    # Assign id to points detected from same pulse
    frame['shot_id'] = gen_shot_ids(frame)
    frame[['shot_id', 'index', 'signal_flag']] = \
        frame[['shot_id', 'index', 'signal_flag']].astype(int)
    frame.filepath = filepath

    return frame


def _load_mabel_gliht_as_frame(filepath, start=None, end=None, channel=43):
    from collections import defaultdict
    import gzip, csv 

    data_dict = defaultdict(list)
    with gzip.open(filepath, 'rb') as csvgz:
        reader = csv.reader(csvgz)
        column_names = reader.next()
        columns = [col.strip() for col in column_names]
        i = 0
        for row in reader:
            if end is not None and i > end:
                break
            if start is None or i >= start:
                row_dict = dict(zip(columns, row))
                if int(float(row_dict['channel'])) == channel:
                    data_dict['d'].append(float(row_dict['atd']))
                    data_dict['z'].append(float(row_dict['elev']))
                i += 1

    data_dict['index'] = arange(len(data_dict['d']))
    frame = pandas.DataFrame(data_dict)
    frame['shot_id'] = gen_shot_ids(frame)
    frame['signal_flag'] = ones(len(frame))
    frame[['shot_id', 'index', 'signal_flag']] = \
        frame[['shot_id', 'index', 'signal_flag']].astype(int)

    return frame


def _load_matlas_gliht_as_frame(filepath, start=None, end=None):
    from collections import defaultdict
    import csv 

    data_dict = defaultdict(list)
    with open(filepath, 'rb') as csv_file:
        reader = csv.reader(csv_file)
        column_names = reader.next()
        columns = [col.strip() for col in column_names]
        i = 0
        for row in reader:
            if end is not None and i > end:
                break
            if start is None or i >= start:
                row_dict = dict(zip(columns, row))
                if int(float(row_dict['channel'])) == channel:
                    data_dict['d'].append(float(row_dict['atd']))
                    data_dict['z'].append(float(row_dict['elev']))
                    data_dict['shot_id'].append(int(row_dict['shot_num']))
                i += 1

    data_dict['index'] = arange(len(data_dict['d']))
    frame = pandas.DataFrame(data_dict)
    frame['shot_id'] = gen_shot_ids(frame)
    frame['signal_flag'] = ones(len(frame))
    frame[['shot_id', 'index', 'signal_flag']] = \
        frame[['shot_id', 'index', 'signal_flag']].astype(int)

    return frame


def forward_filter_backward_sample(kf, y, prior_mu=None, prior_cov=None):
    '''
    Sample from the DLM kf conditioned on observations y.
    '''
    dim_obs = y.shape
    n_obs = dim_obs[0]
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

        if prior_mu is not None:
            m = (V**-1 + prior_cov[i]**-1)**-1*(m*V**-1 + prior_mu[i]*prior_cov[i]**-1)
            V = (V**-1 + prior_cov[i]**-1)**-1

        N = MVNormal(m.flatten(), V)
        x[i] = N.rvs()

    return x

