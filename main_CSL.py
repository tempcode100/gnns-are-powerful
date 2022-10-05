import numpy as np
from collections import defaultdict
from util import load_data_v2
from torch_geometric.datasets import GNNBenchmarkDataset
import networkx as nx


def main():


    dataset = GNNBenchmarkDataset(root='./dataset', name='CSL')

    graphs, num_classes = load_data_v2(dataset)

    dct = defaultdict(list)

    h = [0,1,-1/2, 1/3, -1/4, 1/5, -1/6, 1/7, -1/8, 1/9]
    n_g = len(graphs)
    y = np.zeros((n_g))
    for i in range(n_g):
        G = graphs[i]
        label = G.label
        S = nx.convert_matrix.to_numpy_array(G.g)

        N = S.shape[0]
        tmp = np.eye(N)
        Z = np.zeros((N,N))
        for j in range(len(h)):
            Z += h[j] * tmp
            tmp = tmp @ S

        y1 = np.diagonal(Z)
        y[i] = np.round(np.sum(y1),2)
        if label not in dct:
            dct[label] = [y[i]]
        else:
            if y[i] not in dct[label]:
                dct[label].append(y[i])

    print("GNN output for each class")
    for key, value in dct.items():
        print(key, ' : ', value[0])
    # print(dct)



if __name__ == '__main__':
    main()
