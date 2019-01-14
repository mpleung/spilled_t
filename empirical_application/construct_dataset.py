import numpy as np, networkx as nx, pandas as pd

vil_num = 2

### read data ###

directory = 'cai_data/data/' # directory of data used in cai et al
edgelist = pd.read_stata(directory \
        + '0422allinforawnet.dta')[['id','network_id']].values.astype('int') 
G = nx.DiGraph()
for i in range(edgelist.shape[0]):
    G.add_edge(edgelist[i,0],edgelist[i,1])

data = pd.read_stata(directory + '0422analysis.dta')
data['preintensive'] = data['intensive']*(1-data['delay'])

### construct node-level data matrix ###

# drop all nodes who don't show up in G
drops = np.zeros(data.shape[0])
for i,nid in enumerate(data['id']):
    if not G.has_node(nid): drops[i] == 1
data = data[drops==0]

data2 = data.set_index('village')
villages = np.unique(data['village'].values)
village_sizes = np.zeros(villages.shape[0])
for i,vil in enumerate(villages):
    village_sizes[i] = data[data['village']==vil].shape[0]
sorted_villages = villages[np.argsort(village_sizes)]
village_sizes = village_sizes[np.argsort(village_sizes)]
largest = sorted_villages[(villages.shape[0]-vil_num):villages.shape[0]]
#largest = sorted_villages[0:vil_num]
print '%d villages in data.' % len(villages)
print 'largest %d villages are %s, with total pop %d' % (vil_num, largest, \
        #village_sizes[0:vil_num].sum())
        village_sizes[(villages.shape[0]-vil_num):villages.shape[0]].sum())

data2 = data2.drop([v for v in villages if v not in largest], axis=0) 
tmp = G.copy()
for i in tmp.nodes():
    if i not in data2['id'].values: G.remove_node(i)

# restrict to subset of nodes analyzed by cai et al.
data2 = data2[data2['delay']*data2['info_none']==1]

# write csv
data2 = data2[['id','takeup_survey','intensive','network_rate_preintensive', 'network_obs', 'male','age','agpop','ricearea_2010','literacy','risk_averse','disaster_prob']].dropna()
# network_rate_preintensive = fraction of treated friends, network_obs = out degree

data2.to_csv('data/cai_node_data.csv', header=False, index=True)

### construct dependency graph for network SEs ###

A = G.to_undirected()
for i in G.nodes():
    A.add_edges_from([(i,j) for j in nx.single_source_shortest_path_length(G, i, cutoff=2)])

# restrict to subgraph on nodes in data2
node_list = data2.dropna()['id'].values
tmp = A.copy()
for i in tmp.nodes():
    if i not in node_list: A.remove_node(i)

nx.write_adjlist(A,'data/dep_graph.adjlist')

### count cross-village links ###
data = pd.read_csv('data/cai_node_data.csv', header=None).values
count = 0
for edge in A.edges():
    i = np.where(data[:,1]==int(edge[0]))[0][0]
    j = np.where(data[:,1]==int(edge[1]))[0][0]
    if data[i,0] != data[j,0]: count += 1
print "%d cross-village links" % count
