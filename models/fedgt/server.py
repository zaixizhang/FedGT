import time
import numpy as np
from misc.utils import *
from models.nets import *
from modules.federated import ServerModule
from scipy.spatial.distance import cosine
import multiprocessing

class Server(ServerModule):

    def __init__(self, args, sd, gpu_server):
        super(Server, self).__init__(args, sd, gpu_server)
        self.model = self.model = GT(
            n_layers=self.args.n_layers,
            num_heads=self.args.num_heads,
            input_dim=self.args.n_feat,
            hidden_dim=self.args.hidden_dim,
            output_dim=self.args.n_clss,
            attn_bias_dim=args.attn_bias_dim,
            attention_dropout_rate=args.attention_dropout_rate,
            dropout_rate=args.dropout_rate,
            intput_dropout_rate=args.intput_dropout_rate,
            ffn_dim=args.ffn_dim,
            num_global_node=args.num_global_node,
            num_centroids=args.num_centroids
        ).to(self.gpu_id)
        self.log = {
            'rnd_valid_acc': [], 'rnd_valid_lss': [],
            'rnd_test_acc': [], 'rnd_test_lss': [],
            'best_val_rnd': 0, 'best_val_acc': 0, 'test_acc': 0
        }
        #self.sd['proxy'] = self.get_proxy_data(args.n_feat)
        self.update_lists = []
        self.sim_matrices = []

        n_connected = round(self.args.n_clients*self.args.frac)
        self.avg_sim_matrix = np.zeros(shape=(n_connected, n_connected))
        self.codebook = np.zeros(shape=(self.args.num_centroids, self.args.hidden_dim))

    def get_proxy_data(self, n_feat):
        import networkx as nx
        from torch_geometric.utils import from_networkx

        num_graphs = self.args.n_proxy
        num_nodes = 100
        G = nx.random_partition_graph(
            [num_nodes] * num_graphs, p_in=0.1, p_out=0, seed=self.args.seed)
        data = from_networkx(G)
        data.x = torch.normal(mean=0, std=1, size=(num_nodes * num_graphs, n_feat))

        return data

    def on_round_begin(self, selected, curr_rnd):
        self.round_begin = time.time()
        self.curr_rnd = curr_rnd
        self.sd['global'] = self.get_weights()

    def on_round_complete(self, updated):
        self.update(updated)
        valid_acc, valid_lss = self.validate()
        test_acc, test_lss = self.evaluate()
        
        if self.log['best_val_acc'] <= valid_acc:
            self.log['best_val_rnd'] = self.curr_rnd+1
            self.log['best_val_acc'] = valid_acc
            self.log['test_acc'] = test_acc
            self.save_state()

        self.log['rnd_valid_acc'].append(valid_acc)
        self.log['rnd_valid_lss'].append(valid_lss)
        self.log['rnd_test_acc'].append(test_acc)
        self.log['rnd_test_lss'].append(test_lss)
        self.logger.print(
            f"rnd:{self.curr_rnd+1}, curr_valid_lss:{valid_lss:.4f}, curr_valid_acc:{valid_acc:.4f}, "
            f"best_valid_acc:{self.log['best_val_acc']:.4f}, test_acc:{self.log['test_acc']:.4f} ({time.time()-self.round_begin:.2f}s)"
        )
        self.save_log()

    def update(self, updated):
        st = time.time()
        local_weights = []
        local_codebooks = []
        local_train_sizes = []
        for c_id in updated:
            local_weights.append(self.sd[c_id]['model'].copy())
            local_codebooks.append(self.sd[c_id]['codebook'])
            local_train_sizes.append(self.sd[c_id]['train_size'])
            del self.sd[c_id]
        self.logger.print(f'all clients have been uploaded ({time.time()-st:.2f}s)')

        n_connected = round(self.args.n_clients*self.args.frac)
        assert n_connected == len(local_codebooks)
        sim_matrix = np.empty(shape=(n_connected, n_connected))
        reorder = np.zeros((n_connected, n_connected, self.args.num_centroids), dtype=np.int32)
        for i in range(n_connected):
            for j in range(n_connected):
                sim_matrix[i, j], reorder[i, j] = alignment_distance(local_weights[i], local_weights[j])
        if self.args.agg_norm == 'exp':
            sim_matrix = np.exp(self.args.norm_scale * sim_matrix)

        if self.args.cluster:
            for i in range(n_connected):
                mask = (sim_matrix[i] < sim_matrix[i].mean())
                sim_matrix[i][mask] = 0
        row_sums = sim_matrix.sum(axis=1)
        sim_matrix = sim_matrix / row_sums[:, np.newaxis]
        print(sim_matrix)

        st = time.time()
        ratio = (np.array(local_train_sizes)/np.sum(local_train_sizes)).tolist()
        self.set_weights(self.model, self.aggregate(local_weights, ratio=ratio))
        self.logger.print(f'global model has been updated ({time.time()-st:.2f}s)')

        st = time.time()
        for i, c_id in enumerate(updated):
            local_ratio = sim_matrix[i, :]
            aggr_local_model_weights = self.aggregate(local_weights,  ratio=local_ratio, index=i)
            aggr_local_codebook = self.aggregate_codebook(local_codebooks, ratio=local_ratio, reorder=reorder[i, :])
            if f'adaptive_{c_id}' in self.sd:
                del self.sd[f'adaptive_{c_id}']
            self.sd[f'adaptive_{c_id}'] = {'model': aggr_local_model_weights, 'codebook': aggr_local_codebook}
        self.update_lists.append(updated)
        self.sim_matrices.append(sim_matrix)
        self.logger.print(f'local model has been updated ({time.time()-st:.2f}s)')

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict, self.gpu_id)

    def get_weights(self):
        return {'model': get_state_dict(self.model), 'codebook': self.codebook}

    def save_state(self):
        torch_save(self.args.checkpt_path, 'server_state.pt', {
            'model': get_state_dict(self.model),
            'log': self.log,
            'sim_matrices': self.sim_matrices,
            'update_lists': self.update_lists
        })
