import numpy as np, networkx as nx, math, pandas as pd, os, sys
from functions import *
from scipy import sparse

##########################
### Generate variables ###
##########################

seed = 0
np.random.seed(seed=seed)

B = 1000
network_size = [1000, 2500, 5000]
alpha = 0.95 # desired coverage

p_d = 0.3 # probability of being treated
eff_obs = np.zeros((2,B,len(network_size))) # obs in 'cell'

# dimensions of 3d array read 'z,row,col'
ATE_ests = np.zeros((B,len(network_size))) 
ATE_SE_ests = np.zeros((B,len(network_size)))
ATE_CI_covers = np.zeros((B,len(network_size)))
ATE_SE_failures = np.zeros((len(network_size)))

linear_est = np.zeros((len(network_size),B,4))
linear_SE = np.zeros((len(network_size),B,4))
linear_coverage = np.zeros((len(network_size),B,4))
linear_SE_failures = np.zeros(len(network_size))

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

theta_linear = np.array([1, 2, 1.5, 0.5]) # parameters for linear model

##################
### Estimation ###
##################

for i in range(len(network_size)):
    n = network_size[i]
    print 'n = %d' % n

    for b in range(B):
        # network formation
        N = np.random.poisson(n)
        theta_nf = np.array([-0.2, 0.5, 0.3, 0.9])
        (r, kappa) = gen_r(theta_nf, 2, N)
        positions = np.random.uniform(0,1,(N,2))
        Z = np.random.binomial(1, 0.5, N)
        eps_nf = sparse.triu(np.random.normal(size=(N,N)), 1) * theta_nf[3]
        V_exo = gen_V_exo(Z, eps_nf, theta_nf)
        (D, RGG_minus, RGG_exo) = gen_D(gen_RGG(positions, r), V_exo, theta_nf[2])
        G_temp = gen_G(D, RGG_minus, RGG_exo, V_exo, theta_nf[2], N)
        G = snap_to_nx(G_temp)

        G_sparse = nx.to_scipy_sparse_matrix(G)
        degrees = G_sparse*np.ones(N)
        A = dep_graph(G)

        # random coefficients
        theta = np.vstack([G_sparse*np.random.normal(1,1,N) / (degrees + np.ones(N)*(degrees==0)), np.random.normal(1,1,N), np.random.normal(1,1,N), np.random.normal(-1,1,N), np.random.normal(-1,1,N)]).T

        D = np.random.binomial(1, p_d, N)
        X = DTG_array(G, D)
        Y = responses(X, theta)

        eff_obs[0,b,i] = ind(d, t, g, X).sum()
        eff_obs[1,b,i] = ind(d2, t2, g2, X).sum()

        deg_ind1 = (X[:,2] == g)
        deg_ind2 = (X[:,2] == g2)

        # Linear regression
        nu = np.random.normal(0,1,N) 
        eps = G_sparse*nu / (degrees + np.ones(N)*(degrees==0)) + nu # unobservables in linear outcome equation
        Y_linear = responses_linear(X, eps, theta_linear)
        linear_est[i,b,:],linear_SE[i,b,:],linear_coverage[i,b,:],fail = lin_reg(Y_linear, X, A, theta_linear, alpha)
        linear_SE_failures[i] += fail

        ATE_est = ASF(d, t, g, Y, X) - ASF(d2, t2, g2, Y, X)
        ATE_SE_est,fail = ATE_SE(d, t, g, d2, t2, g2, Y, X, A)
        ATE_SE_failures[i] += fail

        ATE_ests[b,i] = ATE_est
        ATE_SE_ests[b,i] = ATE_SE_est

        q = abs(norm.ppf((1-alpha)/2))
        ATE_CI_covers[b,i] = (ATE_est - q * ATE_SE_est < true_ATE) * (true_ATE < ATE_est + q * ATE_SE_est)

#####################
### Print results ###
#####################

filename = 'results.txt'
if os.path.isfile(filename):
    os.remove(filename)
f = open(filename, 'a')
sys.stdout = f

print 'True ATE: %f' % true_ATE

# ATE table
print '\n\\begin{table}[h]'
print '\centering'
print '\caption{ATE}'
table = pd.DataFrame(np.vstack([eff_obs[0,:,:].mean(axis=0), eff_obs[1,:,:].mean(axis=0), ATE_ests[:,:].mean(axis=0), ATE_SE_ests[:,:].mean(axis=0), ATE_CI_covers[:,:].mean(axis=0), network_size, ATE_SE_failures]))
table.index = ['Cell Count L', 'Cell Count R', 'ATE Est', 'SE', 'Coverage', '$n$', 'Failures']
print table.to_latex(float_format = lambda x: '%1.3f' % x, \
        header=False, escape=False)
print '\end{table}'

# OLS table
print '\n\nLinear model.'
print 'True theta:'
print theta_linear
for i in range(len(network_size)):
    print '\n\\begin{table}[h]'
    print '\centering'
    print '\caption{Linear Regression, $n = ' + str(network_size[i]) + '$.}'
    table = pd.DataFrame(np.vstack([linear_est[i,:,:].mean(axis=0), linear_SE[i,:,:].mean(axis=0), linear_coverage[i,:,:].mean(axis=0)]))
    table.index = ['$\hat\theta$', 'SE', 'Coverage']
    print table.to_latex(float_format = lambda x: '%1.3f' % x, header=False, escape=False)
    print '%d Failures.' % linear_SE_failures[i]
print '\end{table}'

f.close()
