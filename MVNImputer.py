#!/usr/bin/env python
# vim: set fileencoding=utf-8 :
#
# Author:   HIGASHI Koichi
# Created:  2016-07-12
#

import pandas as pd
import numpy as np

def MVNImputer(df, epsilon=1e-5, maxiter=100):
    data = df.values
    n_obs, n_var = data.shape
    mu_init = np.nanmean(data, axis=0)
    sigma_init = np.zeros((n_var, n_var))
    for i in range(n_var):
        for j in range(i, n_var):
            vecs = data[:, [i, j]]
            vecs = vecs[~np.any(np.isnan(vecs), axis=1), :].T
            if len(vecs) > 0:
                cov = np.cov(vecs)
                cov = cov[0, 1]
                sigma_init[i, j] = cov
                sigma_init[j, i] = cov
            else:
                sigma_init[i, j] = 1.0
                sigma_init[j, i] = 1.0
    
    print 'Start EM iteration.'
    pre_mu = mu_init
    pre_sigma = sigma_init
    pre_lik = -np.inf
    for n_iter in range(maxiter):
        # E step
        temp_data = np.copy(data)
        for i in range(n_obs):
            if np.any(np.isnan(temp_data[i, :])):
                nans = np.isnan(temp_data[i, :])
                # conditional distribution of multivariate normal
                offset_mu = np.dot(pre_sigma[nans,:][:,~nans], np.dot(np.linalg.inv(pre_sigma[~nans,:][:,~nans]), (temp_data[i, ~nans] - pre_mu[~nans])[:, np.newaxis]))
                temp_data[i, nans] = pre_mu[nans] + offset_mu
        # M step
        new_mu = np.mean(temp_data, axis=0)
        new_sigma = np.cov(temp_data.T)
        new_lik = -0.5 * n_obs * (np.log(2 * np.pi) + np.log(np.linalg.det(new_sigma)))
        for i in range(n_obs):
            new_lik -= 0.5 * np.dot((temp_data[i, :] - new_mu), np.dot(np.linalg.inv(new_sigma), (temp_data[i, :] - new_mu)[:, np.newaxis]))
        print 'ITER =',n_iter,'\tLog likelihood =',new_lik
        if new_lik - pre_lik < epsilon:
            imputed = temp_data
            break
        pre_mu = new_mu
        pre_sigma = new_sigma
        pre_lik = new_lik
    else:
        imputed = temp_data
    print 'End EM iteration.\n'
    return pd.DataFrame(imputed, index=df.index, columns=df.columns)

if __name__ == '__main__':
    # Test data from [ http://koumurayama.com/koujapanese/missing_data.pdf ]
    d = {'motivations':[3,4,5,2,5,3,2,6,3,3],
            'IQ':[83,85,95,96,103,104,109,112,115,116],
            'aptitude':[np.nan,np.nan,np.nan,np.nan,128,102,111,113,117,133]}
    # true aptitude are: 'aptitude':[93,99,98,103,128,102,111,113,117,133]}
    df = pd.DataFrame(d, index=range(1,11))
    print 'Data...'
    print df,'\n'
    print 'Direct mean...'
    print df.mean(skipna=True),'\n'
    imputed = MVNImputer(df)
    print 'Imputed data...'
    print imputed,'\n'
    print 'ML mean...'
    print imputed.mean(),'\n'
