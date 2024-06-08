import os
import pdb
import time
import torch
import random
import numpy as np

import networkx as nx
import metispy as metis

import torch_geometric
from torch_geometric.data import Data
import torch_geometric.datasets as datasets
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj, dense_to_sparse
import scipy.sparse as sp
from numpy.linalg import inv
import torch.nn.functional as F

from utils import adj_normalize, column_normalize, get_data, split_train

import sys
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from misc.utils import *


mode = 'disjoint' 
data_path = '../../datasets'
seed = 2024
clients = [5,10,20]
ratio_train = 0.2
to_dense = True
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

def generate_data(dataset, n_clients):
    st = time.time()
    data = get_data(dataset, data_path)
    data = split_train(data, dataset, data_path, ratio_train, mode, n_clients)
    split_subgraphs(n_clients, data, dataset)
    print(f'done ({time.time()-st:.2f})')


def split_subgraphs(n_clients, data, dataset):

    G = torch_geometric.utils.to_networkx(data)
    n_cuts, membership = metis.part_graph(G, n_clients)
    assert len(list(set(membership))) == n_clients
    print(f'graph partition done, metis, n_partitions: {len(list(set(membership)))}, n_lost_edges:{n_cuts}')
        
    if to_dense:
        adj = to_dense_adj(data.edge_index)[0]

    for client_id in range(n_clients):
        client_indices = np.where(np.array(membership) == client_id)[0]
        client_indices = list(client_indices)
        client_num_nodes = len(client_indices)

        client_edge_index = []
        if to_dense:
            client_adj = adj[client_indices][:, client_indices]
            client_edge_index, _ = dense_to_sparse(client_adj)
            client_edge_index = client_edge_index.T.tolist()
        else:
            for _index, _edge in enumerate(data.edge_index.T):
                if _edge[0].item() in client_indices and \
                    _edge[1].item() in client_indices:
                    client_edge_index.append([
                        client_indices.index(_edge[0].item()), 
                        client_indices.index(_edge[1].item())
                    ])
        client_num_edges = len(client_edge_index)

        client_edge_index = torch.tensor(client_edge_index, dtype=torch.long)
        client_x = data.x[client_indices]
        client_y = data.y[client_indices]
        client_train_mask = data.train_mask[client_indices]
        client_val_mask = data.val_mask[client_indices]
        client_test_mask = data.test_mask[client_indices]

        client_data = Data(
            x = client_x,
            y = client_y,
            edge_index = client_edge_index.t().contiguous(),
            train_mask = client_train_mask,
            val_mask = client_val_mask,
            test_mask = client_test_mask
        )
        ########################################################################################
        client_adj = sp.coo_matrix((np.ones(client_data.edge_index.shape[1]), (client_data.edge_index[0], client_data.edge_index[1])),
                            shape=(client_data.y.shape[0], client_data.y.shape[0]),
                            dtype=np.float32)
        normalized_adj = adj_normalize(client_adj)
        column_normalized_adj = column_normalize(client_adj)
        power_adj_list = [normalized_adj]
        for m in range(5):
            power_adj_list.append(power_adj_list[0] * power_adj_list[m])
        c = 0.15
        ppr = torch.tensor(c * inv((sp.eye(client_adj.shape[0]) - (1 - c) * column_normalized_adj).toarray()))
        ########################################################################################
        assert torch.sum(client_train_mask).item() > 0

        torch_save(data_path, f'{dataset}_disjoint/{n_clients}/partition_{client_id}.pt', {
            'client_data': client_data,
            'client_id': client_id,
            'ppr': ppr,
            'power_adj_list': power_adj_list,
        })
        print(f'client_id:{client_id}, iid, n_train_node:{client_num_nodes}, n_train_edge:{client_num_edges}')



print('Begin to split graph!')
for n_clients in clients:
    print(n_clients)
    generate_data(dataset='Cora', n_clients=n_clients)
print('Finish splitting')
