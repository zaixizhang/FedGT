import os
import torch
import numpy as np

import torch_geometric.datasets as datasets
import torch_geometric.transforms as T

from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp

from ogb.nodeproppred import PygNodePropPredDataset

def torch_save(base_dir, filename, data):
    os.makedirs(os.path.join(base_dir, os.path.dirname(filename)), exist_ok=True)
    fpath = os.path.join(base_dir, filename)    
    torch.save(data, fpath)

def get_data(dataset, data_path):
    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        data = datasets.Planetoid(data_path, dataset, transform=T.Compose([LargestConnectedComponents(), T.NormalizeFeatures()]))[0]
    elif dataset in ['Computers', 'Photo']:
        data = datasets.Amazon(data_path, dataset, transform=T.Compose([LargestConnectedComponents(), T.NormalizeFeatures()]))[0]
        data.train_mask, data.val_mask, data.test_mask \
            = torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool)
    elif dataset in ['ogbn-arxiv']:
        data = PygNodePropPredDataset(dataset, root=data_path, transform=T.Compose([T.ToUndirected(), LargestConnectedComponents()]))[0]
        data.train_mask, data.val_mask, data.test_mask \
            = torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool)
        data.y = data.y.view(-1)
    return data

def split_train(data, dataset, data_path, ratio_train, mode, n_clients):
    n_data = data.num_nodes
    ratio_test = (1-ratio_train)/2
    n_train = round(n_data * ratio_train)
    n_test = round(n_data * ratio_test)
    
    permuted_indices = torch.randperm(n_data)
    train_indices = permuted_indices[:n_train]
    test_indices = permuted_indices[n_train:n_train+n_test]
    val_indices = permuted_indices[n_train+n_test:]

    data.train_mask.fill_(False)
    data.test_mask.fill_(False)
    data.val_mask.fill_(False)

    data.train_mask[train_indices] = True
    data.test_mask[test_indices] = True
    data.val_mask[val_indices] = True

    torch_save(data_path, f'{dataset}_{mode}/{n_clients}/train.pt', {'data': data})
    torch_save(data_path, f'{dataset}_{mode}/{n_clients}/test.pt', {'data': data})
    torch_save(data_path, f'{dataset}_{mode}/{n_clients}/val.pt', {'data': data})
    print(f'splition done, n_train: {n_train}, n_test: {n_test}, n_val: {len(val_indices)}')
    return data

class LargestConnectedComponents(BaseTransform):
    r"""Selects the subgraph that corresponds to the
    largest connected components in the graph.

    Args:
        num_components (int, optional): Number of largest components to keep
            (default: :obj:`1`)
    """
    def __init__(self, num_components: int = 1):
        self.num_components = num_components

    def __call__(self, data: Data) -> Data:
        import numpy as np
        import scipy.sparse as sp

        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)

        num_components, component = sp.csgraph.connected_components(adj)

        if num_components <= self.num_components:
            return data

        _, count = np.unique(component, return_counts=True)
        subset = np.in1d(component, count.argsort()[-self.num_components:])

        return data.subgraph(torch.from_numpy(subset).to(torch.bool))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_components})'


def adj_normalize(mx):
    "A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2"
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def column_normalize(mx):
    "A' = A * D^-1 "
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = mx.dot(r_mat_inv)
    return mx