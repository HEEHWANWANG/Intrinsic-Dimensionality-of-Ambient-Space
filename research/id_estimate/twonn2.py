# https://github.com/jmmanley/two-nn-dimensionality-estimator/blob/master/twonn.py

import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import tensorflow as tf

#from memory_profiler import profile
import time
import copy
import math

# TWO-NN METHOD FOR ESTIMATING INTRINSIC DIMENSIONALITY
# Facco, E., d’Errico, M., Rodriguez, A., & Laio, A. (2017).
# Estimating the intrinsic dimension of datasets by a minimal neighborhood information.
# Scientific reports, 7(1), 12140.


# Implementation by Jason M. Manley, jmanley@rockefeller.edu
# June 2019
class Time_profile:
    def __init__(self, print_log=True):
        self.print_log = print_log
        self.timestamp = [time.time()]
        if self.print_log:
            print("time_profile initialized")

    def log(self, name=None):
        self.timestamp.append(time.time())
        if len(self.timestamp) >= 2 and self.print_log:
            elapsed = self.timestamp[-1] - self.timestamp[-2]
            print(f'--time_profile {name}: {format(elapsed, ".2f")} sec')

#@profile
def estimate_id_cpu(X, decompose_estimate, plot=False, X_is_dist=False, verbose=False):
    # INPUT:
    #   X = Nxp matrix of N p-dimensional samples (when X_is_dist is False)
    #   plot = Boolean flag of whether to plot fit
    #   X_is_dist = Boolean flag of whether X is an NxN distance metric instead
    #
    # OUTPUT:
    #   d = TWO-NN estimate of intrinsic dimensionality
    tp = Time_profile(print_log = verbose)

    N = X.shape[0]

    if X_is_dist:
        dist = X
    else:
        # COMPUTE PAIRWISE DISTANCES FOR EACH POINT IN THE DATASET
        if len(X.shape) > 2:
            X = X.reshape((X.shape[0], -1))
        if decompose_estimate:
            n = X.shape[0]
            s = n // decompose_estimate
            dist_l = []
            for i in range(decompose_estimate):
                X_decomp = X[s*i:min(s*(i+1), n)]
                dist_l.append(scipy.spatial.distance.cdist(X_decomp, X, metric='euclidean'))
                tp.log(f'cdist({i})')
            dist = np.concatenate(dist_l, axis=0)
        else:
            dist = scipy.spatial.distance.pdist(X, metric='euclidean')
            tp.log(f'pdist')
            dist = scipy.spatial.distance.squareform(dist)
            tp.log(f'squareform')

    # FOR EACH POINT, COMPUTE mu_i = r_2 / r_1,
    # where r_1 and r_2 are first and second shortest distances
    mu = np.zeros(N)

    for i in range(N):
        sort_idx = np.argsort(dist[i,:])
        mu[i] = dist[i,sort_idx[2]] / dist[i,sort_idx[1]]
    tp.log(f'argsort')

    # COMPUTE EMPIRICAL CUMULATE
    sort_idx = np.argsort(mu)
    Femp     = np.arange(N)/N

    # FIT (log(mu_i), -log(1-F(mu_i))) WITH A STRAIGHT LINE THROUGH ORIGIN
    lr = LinearRegression(fit_intercept=False)
    lr.fit(np.log(mu[sort_idx]).reshape(-1,1), -np.log(1-Femp).reshape(-1,1))
    tp.log(f'lr.kfit')

    d = lr.coef_[0][0] # extract slope

    if plot:
        # PLOT FIT THAT ESTIMATES INTRINSIC DIMENSION
        s=plt.scatter(np.log(mu[sort_idx]), -np.log(1-Femp), c='r', label='data')
        p=plt.plot(np.log(mu[sort_idx]), lr.predict(np.log(mu[sort_idx]).reshape(-1,1)), c='k', label='linear fit')
        plt.xlabel('$\log(\mu_i)$'); plt.ylabel('$-\log(1-F_{emp}(\mu_i))$')
        plt.title('ID = ' + str(np.round(d, 3)))
        plt.legend()

    return d


#@profile
def estimate_id_gpu(X, decompose_estimate, verbose=False):
    # INPUT:
    #   X = Nxp matrix of N p-dimensional samples (when X_is_dist is False)
    #
    # OUTPUT:
    #   d = TWO-NN estimate of intrinsic dimensionality
    tp = Time_profile(print_log=verbose)

    N = X.shape[0]
    print("estimate_id size:", X.shape)

    # COMPUTE PAIRWISE DISTANCES FOR EACH POINT IN THE DATASET
    if len(X.shape) > 2:
        X = X.reshape((X.shape[0], -1))

    with tf.device('gpu:0'):
        X = tf.convert_to_tensor(X, dtype=tf.float32)

        s = math.ceil(N / max(decompose_estimate, 1))
        dist_l = np.zeros((N, decompose_estimate*3))

        for i in range(decompose_estimate):
            X_decomp_i = X[s*i:min(s*(i+1), N)]
            for j in range(decompose_estimate):
                X_decomp_j = X[s*j:min(s*(j+1),N)]
                dist = tf.math.sqrt(tf.reduce_sum((tf.expand_dims(X_decomp_i, 1)-tf.expand_dims(X_decomp_j, 0))**2,2)+1e-5)

                dist_sorted, _= tf.math.top_k(-dist,k=3)
                dist_l[s*i:min(s*(i+1),N), 3*j:3*(j+1)] = dist_sorted[:,:3].numpy()

            tp.log(f'dist,sort({i}){X_decomp_i.shape[0]}x{X_decomp_j.shape[0]}')

    mu = np.zeros(N)

    for i in range(N):
        sort_idx = np.argsort(-dist_l[i,:])
        mu[i] = dist_l[i,sort_idx[2]] / dist_l[i,sort_idx[1]]
    tp.log(f'argsort')

    # COMPUTE EMPIRICAL CUMULATE
    sort_idx = np.argsort(mu)
    Femp     = np.arange(N)/N

    # FIT (log(mu_i), -log(1-F(mu_i))) WITH A STRAIGHT LINE THROUGH ORIGIN
    lr = LinearRegression(fit_intercept=False)
    lr.fit(np.log(mu[sort_idx]).reshape(-1,1), -np.log(1-Femp).reshape(-1,1))
    tp.log(f'lr.fit')

    d = lr.coef_[0][0] # extract slope

    return d


def estimate_id(X, gpu, decompose_estimate=1, verbose=False):
    if gpu == False:
        return estimate_id_cpu(X, decompose_estimate, verbose=verbose)
    else:
        return estimate_id_gpu(X, decompose_estimate, verbose=verbose)


def estimate_id_from_dataset_gpu(dataset, model, verbose=False):
    # INPUT:
    #   X = Nxp matrix of N p-dimensional samples (when X_is_dist is False)
    #
    # OUTPUT:
    #   d = TWO-NN estimate of intrinsic dimensionality
    tp = Time_profile(print_log=verbose)

#    print("dataset.cardinality(): ", dataset.cardinality())
#    print("dataset._batch_size: ", dataset._batch_size)

    ds_i = dataset
    ds_j = copy.copy(dataset)

    with tf.device('gpu:0'):
        dist_ij = []
        N = 0

        for i, (b_i, _) in ds_i.enumerate():
            X_i = model.predict(b_i)
            X_i = tf.reshape(X_i, (X_i.shape[0], -1))
            N += X_i.shape[0]

            dist_j = []
            j_length = 0

            for j, (b_j, _) in ds_j.enumerate():
                X_j = model.predict(b_j)
                X_j = tf.reshape(X_j, (X_j.shape[0], -1))
                j_length += X_j.shape[0]

                dist = tf.math.sqrt(tf.reduce_sum((tf.expand_dims(X_i, 1)-tf.expand_dims(X_j, 0))**2,2)+1e-5)
                if dist.shape[1] < 3:
                    dist_sorted = -dist
                else:
                    dist_sorted, _= tf.math.top_k(-dist,k=3)
                dist_j.append(dist_sorted[:,:3].numpy())
#                tp.log(f'dist,sort({i},{j}){X_i.shape[0]}x{X_j.shape[0]}')

            dist_j = tf.concat(dist_j, axis=1)
            dist_ij.append(dist_j)

            tp.log(f'dist,sort({i}){X_i.shape[0]}x{j_length}')
        dist_ij = tf.concat(dist_ij, axis=0)

    mu = np.zeros(N)

    for i in range(N):
        sort_idx = np.argsort(-dist_ij[i,:])
        mu[i] = dist_ij[i,sort_idx[2]] / dist_ij[i,sort_idx[1]]
    tp.log(f'argsort')

    # COMPUTE EMPIRICAL CUMULATE
    sort_idx = np.argsort(mu)
    Femp     = np.arange(N)/N

    # FIT (log(mu_i), -log(1-F(mu_i))) WITH A STRAIGHT LINE THROUGH ORIGIN
    lr = LinearRegression(fit_intercept=False)
    lr.fit(np.log(mu[sort_idx]).reshape(-1,1), -np.log(1-Femp).reshape(-1,1))
    tp.log(f'lr.fit')

    d = lr.coef_[0][0] # extract slope

    return d


def estimate_id_from_dataset(dataset, model, gpu, verbose=False):
    if gpu == False:
        assert 0, "estimate_id_from_dataset for cpu is not implemented yet"
    else:
        return estimate_id_from_dataset_gpu(dataset, model, verbose=verbose)

