import networkx as nx
import numpy as np
import torch
import math
from sklearn.model_selection import StratifiedKFold

###################################################################################################################################

def compute_diagonal(S):
    n = S.shape[0]
    diag = torch.zeros(n)
    for i in range(n):
        diag[i] = S[i,i] 
    return diag

def add_attributes_diag_norm(graph_list, K):
    for i,graph in enumerate(graph_list):
        N = graph.node_features.shape[0]
        diag_k = torch.zeros(N,K)
        diag_k[:,0] = torch.ones(N)
        Adj_idx = graph.edge_mat
        Adj_elem = torch.ones(Adj_idx.shape[1])
        S = torch.sparse.FloatTensor(Adj_idx, Adj_elem, torch.Size([N,N]))
        x = S
        diag_k[:,1] = compute_diagonal(x)
        for k in range(2,K):
            x = torch.sparse.mm(S, x)
            diag_k[:,k] = compute_diagonal(x)
        y = diag_k
        mn = torch.mean(y,1).reshape(y.shape[0],1)
        y = y - mn.expand_as(y)
        norm_y = torch.sum(torch.square(y),1).reshape(y.shape[0],1)
        norm_y[norm_y==0] = 1
        y = y / norm_y
        graph_list[i].node_features = y
    return graph_list

def add_attributes_norm(graph_list, K):
    for i,graph in enumerate(graph_list):
        N = graph.node_features.shape[0]
        deg_k = torch.zeros(N,K)
        diag_k = torch.zeros(N,K)
        Adj_idx = graph.edge_mat
        Adj_elem = torch.ones(Adj_idx.shape[1])
        S = torch.sparse.FloatTensor(Adj_idx, Adj_elem, torch.Size([N,N]))
        x = S
        for k in range(K):
            deg_k[:,k] = torch.sparse.sum(x,1).to_dense()
            x = torch.sparse.mm(S, x)/math.factorial(k+1)
            if k > 0:
                diag_k[:,k-1] = compute_diagonal(x)/math.factorial(k+1)
        y = torch.cat((diag_k, deg_k), 1)
        mn = torch.mean(y,1).reshape(y.shape[0],1)
        y = y - mn.expand_as(y)
        norm_y = torch.sum(torch.square(y),1).reshape(y.shape[0],1)
        norm_y[norm_y==0] = 1
        y = y / norm_y
        
        graph_list[i].node_features = y
    return graph_list

def add_attributes_diag(graph_list, K):
    for i,graph in enumerate(graph_list):
        N = graph.node_features.shape[0]
        diag_k = torch.zeros(N,K)
        diag_k[:,0] = torch.ones(N)
        Adj_idx = graph.edge_mat
        Adj_elem = torch.ones(Adj_idx.shape[1])
        S = torch.sparse.FloatTensor(Adj_idx, Adj_elem, torch.Size([N,N]))
        x = S
        diag_k[:,1] = compute_diagonal(x)
        for k in range(2,K):
            x = torch.sparse.mm(S, x)
            diag_k[:,k] = compute_diagonal(x)
        y = diag_k
        graph_list[i].node_features = y
    return graph_list

def add_attributes(graph_list, K):
    for i,graph in enumerate(graph_list):
        N = graph.node_features.shape[0]
        deg_k = torch.zeros(N,K)
        diag_k = torch.zeros(N,K)
        Adj_idx = graph.edge_mat
        Adj_elem = torch.ones(Adj_idx.shape[1])
        S = torch.sparse.FloatTensor(Adj_idx, Adj_elem, torch.Size([N,N]))
        x = S
        for k in range(K):
            deg_k[:,k] = torch.sparse.sum(x,1).to_dense()
            x = torch.sparse.mm(S, x)/math.factorial(k+1)
            if k > 0:
                diag_k[:,k-1] = compute_diagonal(x)/math.factorial(k+1)
        y = torch.cat((diag_k, deg_k), 1)
        graph_list[i].node_features = y
    return graph_list

def add_ones(graph_list):
    for i,graph in enumerate(graph_list):
        N = graph.node_features.shape[0]
        y = torch.ones(N,1)
        graph_list[i].node_features = y
    return graph_list

###################################################################################################################################

###################################################################################################################################
## The functions in this box have been modified from https://github.com/weihua916/powerful-gnns (Copyright (c) 2021 Weihua Hu)#####

class S2VGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.node_tags = node_tags
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_data(dataset):
    '''
        dataset: name of dataset
        test_proportion: ratio of test train split
        seed: random seed for random splitting of dataset
    '''

    print('loading data')
    g_list = []
    label_dict = {}
    feat_dict = {}

    with open('dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
        n_g = int(f.readline().strip())
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]
            if not l in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.Graph()
            node_tags = []
            node_features = []
            n_edges = 0
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                tmp = int(row[1]) + 2
                if tmp == len(row):
                    # no node attributes
                    row = [int(w) for w in row]
                    attr = None
                else:
                    row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
                if not row[0] in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[row[0]] = mapped
                node_tags.append(feat_dict[row[0]])

                if tmp > len(row):
                    node_features.append(attr)

                n_edges += row[1]
                for k in range(2, len(row)):
                    g.add_edge(j, row[k])

            if node_features != []:
                node_features = np.stack(node_features)
            else:
                node_features = None

            assert len(g) == n

            g_list.append(S2VGraph(g, l, node_tags))

    #add labels and edge_mat       
    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    #Extracting unique tag labels   
    tagset = set([])
    for g in g_list:
        tagset = tagset.union(set(g.node_tags))

    tagset = list(tagset)
    tag2index = {tagset[i]:i for i in range(len(tagset))}

    for g in g_list:
        g.node_features = torch.zeros(len(g.node_tags), len(tagset))
        g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


    print('# classes: %d' % len(label_dict))
    print('# maximum node tag: %d' % len(tagset))

    print("# data: %d" % len(g_list))

    return g_list, len(label_dict)

def load_data_v2(dataset):

    g_list = []
    n_g = dataset.data.y.shape[0]
    for i in range(n_g):
        g = nx.Graph()
        l = int(dataset.data.y[i])
        graph = dataset[i]
        n_e = graph.edge_index.shape[1]
        for j in range(n_e):
            n1 = int(graph.edge_index[0][j])
            n2 = int(graph.edge_index[1][j])
            g.add_edge(n1, n2)

        g_list.append(S2VGraph(g, l))


    for g in g_list:
        g.neighbors = [[] for i in range(len(g.g))]
        for i, j in g.g.edges():
            g.neighbors[i].append(j)
            g.neighbors[j].append(i)
        degree_list = []
        for i in range(len(g.g)):
            g.neighbors[i] = g.neighbors[i]
            degree_list.append(len(g.neighbors[i]))
        g.max_neighbor = max(degree_list)

        # g.label = label_dict[g.label]

        edges = [list(pair) for pair in g.g.edges()]
        edges.extend([[i, j] for j, i in edges])

        g.edge_mat = torch.LongTensor(edges).transpose(0,1)

    print('# classes: %d' % dataset.num_classes)

    print("# data: %d" % len(g_list))

    return g_list, dataset.num_classes

def pass_data_iteratively(model, graphs, minibatch_size = 1):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)

def separate_data(graph_list, seed, fold_idx):
    assert 0 <= fold_idx and fold_idx < 10, "fold_idx must be from 0 to 9."
    skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = seed)
    # skf = StratifiedShuffleSplit(n_splits=10, test_size = 0.2, random_state = seed)

    labels = [graph.label for graph in graph_list]
    idx_list = []
    for idx in skf.split(np.zeros(len(labels)), labels):
        idx_list.append(idx)
    train_idx, test_idx = idx_list[fold_idx]

    train_graph_list = [graph_list[i] for i in train_idx]
    test_graph_list = [graph_list[i] for i in test_idx]

    return train_graph_list, test_graph_list


def create_norm_block_adj( batch_graph, device):
        ###create block diagonal sparse matrix

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = 1 / Adj_block_idx.shape[1] * torch.ones(Adj_block_idx.shape[1])
        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1],start_idx[-1]]))

        return Adj_block.to(device)

def create_block_adj( batch_graph, device):
        ###create block diagonal sparse matrix

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])
        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1],start_idx[-1]]))

        return Adj_block.to(device)


def create_block_adj_2( batch_graph, device):
        ###create block diagonal sparse matrix

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        #  #Add self-loops in the adjacency matrix if learn_eps is False, i.e., aggregate center nodes and neighbor nodes altogether.
        # if not self.learn_eps:
        #     num_node = start_idx[-1]
        #     self_loop_edge = torch.LongTensor([range(num_node), range(num_node)])
        #     elem = torch.ones(num_node)
        #     Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
        #     Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1],start_idx[-1]]))


        return Adj_block.to(device)


def graphpool_operations( batch_graph, device):
        ###create sum or average pooling sparse matrix over entire nodes in each graph (num graphs x num nodes)
        
        start_idx = [0]

        #compute the padded neighbor list
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.g))

        idx = []
        elem = []
        for i, graph in enumerate(batch_graph):
            ## sum pooling
            elem.extend([1]*len(graph.g))

            idx.extend([[i, j] for j in range(start_idx[i], start_idx[i+1], 1)])
            
        elem = torch.FloatTensor(elem)
        idx = torch.LongTensor(idx).transpose(0,1)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([len(batch_graph), start_idx[-1]]))
        
        return graph_pool.to(device)
###################################################################################################################################
