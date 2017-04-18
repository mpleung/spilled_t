import numpy as np, networkx as nx, math, snap, os, scipy.special
from scipy import spatial, sparse
from scipy.stats import norm
from scipy.special import binom

###################
### Network DGP ###
###################

# Most of these functions are taken from Leung (2017), 'A Weak Law for Moments of Pairwise-Stable Networks.'

def RGG(positions, r):
    """ 
    Returns an RGG (networkx object) from a given n-vector of dx1 positions (nxd matrix). 
    
    positions = vector of node positions.
    r = linking threshold.
    """
    kdtree = spatial.KDTree(positions)
    pairs = kdtree.query_pairs(r) # default is Euclidean norm
    RGG = nx.Graph()
    RGG.add_nodes_from(range(len(positions)))
    for edge in (i for i in list(pairs)):
        RGG.add_edge(edge[0],edge[1])
    return RGG

def gen_RGG(positions, r):
    """ 
    Returns an RGG (snap object) from a given N-vector of dx1 positions (Nxd matrix). 
    
    positions = vector of node positions.
    r = linking threshold.
    """
    kdtree = spatial.KDTree(positions)
    pairs = kdtree.query_pairs(r) # default is Euclidean norm
    RGG = snap.GenRndGnm(snap.PUNGraph, len(positions), 0)
    for edge in (i for i in list(pairs)):
        RGG.AddEdge(edge[0],edge[1])
    return RGG

def copy_graph(graph):
    """
    Returns a copy of a snap network.
    
    Credit: https://stackoverflow.com/questions/23133372/how-to-copy-a-graph-object-in-snap-py
    """
    tmpfile = '.copy.bin'
    
    # Saving to tmp file
    FOut = snap.TFOut(tmpfile)
    graph.Save(FOut)
    FOut.Flush()
    
    # Loading to new graph
    FIn = snap.TFIn(tmpfile)
    graphtype = type(graph)
    new_graph = graphtype.New()
    new_graph = new_graph.Load(FIn)
    
    os.remove(tmpfile)
    
    return new_graph

def ball_vol(d,r):
    """
    Returns the volume of a d-dimensional ball of radius r.
    """
    return math.pi**(d/2) * float(r)**d / scipy.special.gamma(d/2+1)


def gen_r(theta, d, N):
    """
    Returns the RGG threshold r.

    theta = true parameter.
    d = dimension of node positions.
    N = number of nodes.
    """
    vol = ball_vol(d,1)

    Phi2 = norm.cdf(-(theta[0] + 2*theta[1])/theta[3]) - norm.cdf(-(theta[0] + 2*theta[1] + theta[2])/theta[3])
    Phi1 = norm.cdf(-(theta[0] + theta[1])/theta[3]) - norm.cdf(-(theta[0] + theta[1] + theta[2])/theta[3])
    Phi0 = norm.cdf(-theta[0]/theta[3]) - norm.cdf(-(theta[0] + theta[2])/theta[3])
    gamma = math.sqrt( Phi2**2*0.25 + Phi1**2*0.5 + Phi0**2*0.25)
    
    # kappa = limit of nr^d
    kappa = 1/(vol*gamma) - 0.8
    
    # r = (kappa/n)^(1/d)
    return ((kappa/N)**(1/float(d)), kappa)


def gen_V_exo(Z, eps, theta):
    """ 
    Returns 'exogenous' part of joint surplus function for each pair of nodes as a
    sparse upper triangular matrix.

    NB: This function is specific to the joint surplus function used in our
    simulations. 

    eps = sparse NxN upper triangular matrix. 
    Z = N-vector of binary attributes. 
    """
    N = Z.shape[0]
    sparse_ones = sparse.triu(np.ones((N,N)),1)
    Z_sum = sparse.triu(np.tile(Z, (N,1)) + np.tile(Z[:,np.newaxis], N),1)
    U = theta[0] * sparse_ones + theta[1] * Z_sum + eps
    return U

def gen_D(Pi, V_exo, theta2):
    """
    Returns a triplet of three snap graphs:
    D = opportunity graph with robust links removed.
    Pi_minus = subgraph of Pi without robustly absent potential links.
    Pi_exo = subgraph of Pi with only robust links.

    NB: This function is specific to the joint surplus used in our simulations.

    Pi = opportunity graph (in our case, the output of gen_RGG).
    V_exo = 'exogenous' part of joint surplus (output of gen_V_exo).
    theta2 = transitivity parameter (theta[2]).
    """
    N = V_exo.shape[0]
    D = copy_graph(Pi)
    Pi_minus = copy_graph(Pi)
    Pi_exo = snap.GenRndGnm(snap.PUNGraph, N, 0) 

    for edge in Pi.Edges():
        i = min(edge.GetSrcNId(), edge.GetDstNId())
        j = max(edge.GetSrcNId(), edge.GetDstNId())
        if V_exo[i,j] + min(theta2,0) > 0:
            D.DelEdge(i,j) 
            Pi_exo.AddEdge(i,j)
        if V_exo[i,j] + max(theta2,0) <= 0:
            D.DelEdge(i,j)
            Pi_minus.DelEdge(i,j)
 
    return (D, Pi_minus, Pi_exo)

def gen_G_subgraph(component, D, Pi_minus, Pi_exo, V_exo, theta2):
    """ 
    Returns a pairwise-stable network for nodes in component, via myopic best-
    response dynamics. This subnetwork is pairwise-stable taking as given the
    links in the rest of the network. Initial network for best-response dynamics
    is the opportunity graph. 

    NB: This function is specific to the joint surplus used in our simulations.
    
    component = component of D for which we want a pairwise-stable subnetwork.
    D, Pi_minus, Pi_exo = outputs of gen_D().
    V_exo = 'exogenous' part of joint surplus (output of gen_V_exo).
    theta2 = transitivity parameter (theta[2]).
    """
    stable = False
    meetings_without_deviations = 0

    D_subgraph = snap.GetSubGraph(D, component)

    # Start initial network on Pi, without robustly absent potential links.
    G = snap.GetSubGraph(Pi_minus, component)
    
    # For each node pair (i,j) linked in Pi_exo (i.e. their links are robust),
    # with either i or j in component, add their link to G.  Result is the
    # subgraph of Pi_minus on an augmented component of D.
    for i in component:
        for j in Pi_exo.GetNI(i).GetOutEdges():
            if not G.IsNode(j): G.AddNode(j)
            G.AddEdge(i,j)

    while not stable:
        # Need only iterate through links of D, since all other links are
        # robust.
        for edge in D_subgraph.Edges():
            # Iterate deterministically through default edge order order. Add or
            # remove link in G according to myopic best-respnose dynamics. If we
            # cycle back to any edge with no changes to the network, conclude
            # it's pairwise stable.
            i = min(edge.GetSrcNId(), edge.GetDstNId())
            j = max(edge.GetSrcNId(), edge.GetDstNId())
            cfriend = snap.GetCmnNbrs(G, i, j) > 0
            if V_exo[i,j] + theta2*cfriend > 0: # specific to model of V
                if G.IsEdge(i,j):
                    meetings_without_deviations += 1
                else:
                    G.AddEdge(i,j)
                    meetings_without_deviations = 0
            else:
                if G.IsEdge(i,j):
                    G.DelEdge(i,j)
                    meetings_without_deviations = 0
                else:
                    meetings_without_deviations += 1

        if meetings_without_deviations > D_subgraph.GetEdges():
            stable = True

    return snap.GetSubGraph(G, component)

def gen_G(D, Pi_minus, Pi_exo, V_exo, theta2, N):
    """
    Returns pairwise-stable network on N nodes. 
    
    D, Pi_minus, Pi_exo = outputs of gen_D().
    V_exo = 'exogenous' part of joint surplus (output of gen_V_exo).
    theta2 = transitivity parameter (theta[2]).
    """
    G = snap.GenRndGnm(snap.PUNGraph, N, 0) # initialize empty graph
    Components = snap.TCnComV()
    snap.GetWccs(D, Components) # collects components of D
    NIdV = snap.TIntV() # initialize vector
    for C in Components:
        if C.Len() > 1:
            NIdV.Clr()
            for i in C:
                NIdV.Add(i)
            tempnet = gen_G_subgraph(NIdV, D, Pi_minus, Pi_exo, V_exo, theta2)
            for edge in tempnet.Edges():
                G.AddEdge(edge.GetSrcNId(), edge.GetDstNId())
 
    # add robust links
    for edge in Pi_exo.Edges():
        G.AddEdge(edge.GetSrcNId(), edge.GetDstNId())

    return G

def snap_to_nx(A):
    """ 
    Converts snap object to networkx object.
    """
    G = nx.Graph()
    G.add_nodes_from(range(A.GetNodes()))
    for edge in A.Edges():
        G.add_edge(edge.GetSrcNId(), edge.GetDstNId())
    return G

###################
### Outcome DGP ###
###################

def DTG_array(G, D):
    """
    Returns nx3 matrix, with row i corresponding to the treatment status,
    number of treated neighbors, and number of neighbors of node i.

    G = networkx graph object with nodes labeled 0, ..., n-1.
    D = 1xn treatment assignment vector.
    """
    num_nhbrs = nx.degree(G).values()
    num_nhbrs_treated = [D[G.neighbors(i)].sum() for i in G.nodes()]
    return np.vstack([D, num_nhbrs_treated, num_nhbrs]).T.astype('float')

def responses(X, theta):
    """
    Returns 1xn array of treatment responses according to the first model given in the monte carlo section of the paper.

    theta = nx5, each row a vector of random coefficients.
    X = output of DTG_array().
    """
    tmp = X.copy()
    tmp[X[:,2]==0,1] = 0
    tmp[X[:,2]==0,2] = 1
    Z = np.vstack([np.ones(len(X)), X[:,0], X[:,1], X[:,1]**2, X[:,1]*X[:,2]]).T
    return np.diag(np.dot(Z, theta.T))

def responses_linear(X, eps, theta):
    """
    Returns 1xn array of treatment responses according to the second (linear) model given in the monte carlo section of the paper.

    theta = nx4.
    eps = nx1 vector of errors.
    X = output of DTG_array().
    """
    Z = np.hstack([np.ones(len(X))[:,np.newaxis], X])
    return np.dot(Z, theta) + eps

##################
### ESTIMATION ###
##################

def ind(d,t,g,X):
    """
    Returns 1xn vector with ith entry = 1{X[i,:] = (d,t,g)}.
    """
    return ( (X[:,0] == d) * (X[:,1] == t) * (X[:,2] == g) ).astype('float')
    
def dep_graph(G):
    """
    Returns nxn dependency graph A defined in the paper in the form of a sparse matrix.

    G = networkx object.
    """
    G_sparse = nx.to_scipy_sparse_matrix(G)
    return ( (sparse.identity(G.number_of_nodes()) + G_sparse + sparse.csr_matrix.dot(G_sparse,G_sparse)) > 0 ).astype('float')

def lin_reg(Y,X,A,theta,alpha):
    """
    Returns estimate, standard error, whether or not the truth is covered by the CI for the linear regression model, and indicator for whether or not standard variance formula delivers negative SEs.

    Y = output of responses_linear().
    X = output of DTG_array().
    A = dependency graph.
    theta = true coefficients.
    1-alpha = desired coverage.
    """
    fail = 0
    Z = np.hstack([np.ones(len(X))[:,np.newaxis], X])
    P = np.linalg.inv(Z.T.dot(Z))
    est = P.dot(Z.T.dot(Y))

    resid = Y - Z.dot(est)
    M = Z * resid[:,np.newaxis]
    S = M.T.dot(sparse.csr_matrix.dot(A,M))
    var = np.diag(P.dot(S).dot(P))
    if (var < 0).sum() > 0: 
        fail += 1
        degrees = np.squeeze(np.asarray(A.sum(axis=1)))
        B = A + sparse.spdiags(degrees, 0, degrees.size, degrees.size) - sparse.identity(A.shape[0])
        S = M.T.dot(sparse.csr_matrix.dot(B,M))
        var = np.diag(P.dot(S).dot(P))
    SE = np.sqrt(var)
    
    q = abs(norm.ppf((1-alpha)/2))
    cover = (est - q * SE < theta) * (theta < est + q * SE)
    return est,SE,cover,fail

def ASF(d, t, g, Y, X):
    """
    Returns frequency estimator of ASF evaluated at (d,t,g) where d is
    the treatment assignment, t the number of treated neighbors, and g the
    number of neighbors.

    d = ego treatment assignment (binary).
    t = number of treated friends.
    g = number of friends.
    Y = output of responses().
    X = output of DTG_array().
    """
    Ind = ind(d,t,g,X)
    if Ind.sum() == 0:
        return 0
    else:
        return np.dot(Y, Ind) / Ind.sum()

def ATE_SE(d1, t1, g1, d2, t2, g2, Y, X, A):
    """
    Returns (SE, fail).
    SE = frequency estimate of the standard error of the ATE estimator $\hat\mu(d1,t1,g1-t1) - \hat\mu(d2,t2,g2-t2)$. 
    fail = binary variable for whether or not the standard variance formula is negative.
    """
    fail = 0
    n = float(Y.shape[0])
    ASF1 = ASF(d1, t1, g1, Y, X)
    ASF2 = ASF(d2, t2, g2, Y, X)
    Ind1 = ind(d1,t1,g1,X)
    Ind2 = ind(d2,t2,g2,X)
    P1 = Ind1.sum()
    P2 = Ind2.sum()
    
    if P1 == 0 or P2 == 0:
        SE = 111 # no observations in cell
        fail += 1
    else:
        v = Y * (Ind1 / P1 - Ind2 / P2) - ( ASF1 / P1 * Ind1 - ASF2 / P2 * Ind2 )
        var = np.dot(v, A.dot(v))
        if var < 0:
            fail += 1
            degrees = np.squeeze(np.asarray(A.sum(axis=1)))
            B = A + sparse.spdiags(degrees, 0, degrees.size, degrees.size) - sparse.identity(A.shape[0])
            var = np.dot(v, B.dot(v))
        SE = math.sqrt(var)

    return SE, fail
