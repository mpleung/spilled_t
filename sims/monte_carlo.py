import numpy as np, networkx as nx, math, pandas as pd, sys, os
from functions import *
from scipy import sparse

seed = 0
np.random.seed(seed=seed)

# output 
filename = 'results.txt'
if os.path.isfile(filename): os.remove(filename)
f = open(filename, 'a')
sys.stdout = f

#############
### Setup ###
#############

B = 3000
network_size = [500, 1000, 2000, 3000]
alpha = 0.95 # desired coverage

p_d = 0.3 # probability of being treated
eff_obs = np.zeros((2,B,len(network_size))) # obs in 'cell'

# dimensions of 3d array read 'z,row,col'
ATE_ests = np.zeros((B,len(network_size))) 
ATE_SE_ests = np.zeros((B,len(network_size)))
ATE_CI_covers = np.zeros((B,len(network_size)))
ATE_SE_failures = np.zeros((len(network_size)))

linear_est = np.zeros((len(network_size),B,3))
linear_SE = np.zeros((len(network_size),B,3))
linear_coverage = np.zeros((len(network_size),B,3))
linear_SE_failures = np.zeros(len(network_size))
linear_SE_naive = np.zeros((len(network_size),B,3))
linear_coverage_naive = np.zeros((len(network_size),B,3))

# parameters of ASF: own treatment, # treated neighbors, # neighbors
d = 0
t = 1
g = 3
d2 = 0
t2 = 0
g2 = 3
if t > g or t2 > g2:
    raise ValueError('# treated neighbors > # neighbors')

true_ATE = (1 + d + t - t**2 - t*g) - (1 + d2 + t2 - t2**2 - t2*g2)

theta_nf = np.array([-0.25, 0.5, 0.25, 1])
theta_linear = np.array([-0.25, 0.5, 0.25]) # parameters for linear model
kappa = gen_kappa(theta_nf, 2)

##################
### Estimation ###
##################

for i in range(len(network_size)):
    n = network_size[i]
    print 'n = %d' % n

    for b in range(B):
        # network formation
        r = (kappa/n)**(1/float(2))
        G_temp = gen_SNF(theta_nf, 2, n, r)
        G = snap_to_nx(G_temp)
        G_sparse = nx.to_scipy_sparse_matrix(G)
        degrees = G_sparse*np.ones(n)

        # random coefficients model
        theta = np.vstack([G_sparse*np.random.normal(1,1,n) / (degrees + np.ones(n)*(degrees==0)), np.random.normal(1,1,n), np.random.normal(1,1,n), np.random.normal(-1,1,n), np.random.normal(-1,1,n)]).T
        D = np.random.binomial(1, p_d, n)
        X = DTG_array(G, D)
        Y = responses(X, theta)

        A = dep_graph(G)
        eff_obs[0,b,i] = ind(d, t, g, X).sum()
        eff_obs[1,b,i] = ind(d2, t2, g2, X).sum()

        ATE_est = ASF(d, t, g, Y, X) - ASF(d2, t2, g2, Y, X)
        ATE_SE_est,fail = ATE_SE(d, t, g, d2, t2, g2, Y, X, A)
        ATE_SE_failures[i] += fail

        ATE_ests[b,i] = ATE_est
        ATE_SE_ests[b,i] = ATE_SE_est

        q = abs(norm.ppf((1-alpha)/2))
        ATE_CI_covers[b,i] = (ATE_est - q * ATE_SE_est < true_ATE) * (true_ATE < ATE_est + q * ATE_SE_est)

        # linear model 
        nu = np.random.normal(0,1,n) 
        eps = G_sparse*nu / (degrees + np.ones(n)*(degrees==0)) + nu # unobservables in linear outcome equation
        X_lin = np.vstack([X[:,0], X[:,1]/X[:,2]]).T
        X_lin[np.isnan(X_lin)] = 0
        Y_linear = responses_linear(X_lin, eps, theta_linear)
        linear_est[i,b,:],linear_SE[i,b,:],linear_coverage[i,b,:],fail = lin_reg(Y_linear, X_lin, A, theta_linear, alpha)
        linear_SE_failures[i] += fail
        est,linear_SE_naive[i,b,:],linear_coverage_naive[i,b,:],fail = lin_reg(Y_linear, X_lin, sparse.eye(X.shape[0]).tocsr(), theta_linear, alpha)

#####################
### Print results ###
#####################

print 'True Average Spillover Effect: %f' % true_ATE

# ATE table
print '\n\\begin{table}[h]'
print '\centering'
print '\caption{Nonparametric Estimator}'
table = pd.DataFrame(np.vstack([network_size, eff_obs[0,:,:].mean(axis=0), eff_obs[1,:,:].mean(axis=0), ATE_ests[:,:].mean(axis=0), ATE_SE_ests[:,:].mean(axis=0), ATE_CI_covers[:,:].mean(axis=0), ATE_SE_failures]))
table.index = ['$n$', 'Eff.\ $n$ $(0,1,3)$', 'Eff.\ $n$ $(0,0,3)$', 'Estimate', 'SE', 'Coverage', 'Failures']
print table.to_latex(float_format = lambda x: '%1.2f' % x, \
                header=False, escape=False)
print '\end{table}'

# OLS table
print '\n\nLinear Regression.'
print 'True theta:'
print theta_linear
for i in range(len(network_size)):
    print '\n\\begin{table}[h]'
    print '\centering'
    print '\caption{Linear Regression, $n = ' + str(network_size[i]) + '$.}'
    table = pd.DataFrame(np.vstack([linear_est[i,:,:].mean(axis=0), linear_SE[i,:,:].mean(axis=0), linear_coverage[i,:,:].mean(axis=0), linear_SE_naive[i,:,:].mean(axis=0), linear_coverage_naive[i,:,:].mean(axis=0)]))
    table.index = ['Estimate', 'Network SE', 'Network Coverage', 'HC SE', 'HC Coverage']
    print table.to_latex(float_format = lambda x: '%1.2f' % x, header=False, escape=False)
    print '%d Failures.' % linear_SE_failures[i]
print '\end{table}'

f.close()
