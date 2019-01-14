import numpy as np, networkx as nx, pandas as pd, sys
sys.path.append('../sims/') # directory of functions.py used in simulation study
from scipy import sparse
from functions import *

alpha = 0.95

# output 
filename = 'results.txt'
if os.path.isfile(filename): os.remove(filename)
f = open(filename, 'a')
sys.stdout = f

### Read data ###

data = pd.read_csv('data/cai_node_data.csv', header=None).values
data = data[np.argsort(data[:,0])]
Y = data[:,2].astype('float')
main_regressors = np.vstack([data[:,3], data[:,4]]).T # treatment, fraction of treated friends
friend_counts = np.vstack([data[:,5]==2, data[:,5]==3, data[:,5]==4, data[:,5]==5]).T # dummies for # of friends
data_controls = data[:,6:data.shape[1]] # cai et al controls
X = np.hstack([main_regressors, friend_counts, data_controls])
X = X.astype('float')

# dependence graph 
A_graph = nx.read_adjlist('data/dep_graph.adjlist', create_using=nx.Graph())
A = sparse.lil_matrix((X.shape[0],X.shape[0]))
for edge in A_graph.edges():
    i = np.where(data[:,1]==int(edge[0]))[0][0]
    j = np.where(data[:,1]==int(edge[1]))[0][0]
    A[i,j] = 1
    A[j,i] = 1
A = A.tocsr()

print '%d edges in dependency graph' % A_graph.number_of_edges()
print '%d ones in A' % A.sum()
print 'sample size %d' % data.shape[0]
print 'significance level %f' % alpha

# estimation with correct SEs
bhat,SE,coverage,fail = lin_reg(Y, X, A, np.zeros(X.shape[1]+1), alpha)
print '\n\\begin{table}[h]'
print '\centering'
print '\caption{Network-Robust SEs.}'
table = pd.DataFrame(np.vstack([bhat[1:3], SE[1:3], coverage[1:3]]))
table.index = ['$\hat\theta$', 'SE', 'Covers 0?']
print table.to_latex(float_format = lambda x: '%1.3f' % x, header=False, escape=False)
print '\end{table}'

# estimation with HC SEs
bhat,SE,coverage,fail = lin_reg(Y, X, sparse.eye(X.shape[0]).tocsr(), np.zeros(X.shape[1]+1), alpha)
print '\n\\begin{table}[h]'
print '\centering'
print '\caption{HC SEs.}'
table = pd.DataFrame(np.vstack([bhat[1:3], SE[1:3], coverage[1:3]]))
table.index = ['$\hat\theta$', 'SE', 'Covers 0?']
print table.to_latex(float_format = lambda x: '%1.3f' % x, header=False, escape=False)
print '\end{table}'

f.close()

