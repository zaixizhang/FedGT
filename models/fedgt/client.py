import time
import copy
import torch
import numpy as np
import torch.nn.functional as F

from misc.utils import *
from models.nets import *
from modules.federated import ClientModule

class Client(ClientModule):

    def __init__(self, args, w_id, g_id, sd):
        super(Client, self).__init__(args, w_id, g_id, sd)
        self.model = GT(
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
        ).to(g_id)
        self.parameters = list(self.model.parameters())

    def init_state(self):
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.args.base_lr, weight_decay=self.args.weight_decay)
        self.log = {
            'lr': [],'train_lss': [],
            'ep_local_val_lss': [],'ep_local_val_acc': [],
            'rnd_local_val_lss': [],'rnd_local_val_acc': [],
            'ep_local_test_lss': [],'ep_local_test_acc': [],
            'rnd_local_test_lss': [],'rnd_local_test_acc': [],
        }

    def save_state(self):
        torch_save(self.args.checkpt_path, f'{self.client_id}_state.pt', {
            'optimizer': self.optimizer.state_dict(),
            'model': get_state_dict(self.model),
            'log': self.log,
            'node_feature': self.node_feature,
        })

    def load_state(self):
        loaded = torch_load(self.args.checkpt_path, f'{self.client_id}_state.pt')
        set_state_dict(self.model, loaded['model'], self.gpu_id)
        self.optimizer.load_state_dict(loaded['optimizer'])
        self.log = loaded['log']
    
    def on_receive_message(self, curr_rnd):
        self.curr_rnd = curr_rnd
        self.update(self.sd[f'adaptive_{self.client_id}' \
            if (f'adaptive_{self.client_id}' in self.sd) else 'global'])
        self.global_w = convert_np_to_tensor(self.sd['global']['model'], self.gpu_id)
        self.codebook = self.sd[f'adaptive_{self.client_id}' \
            if (f'adaptive_{self.client_id}' in self.sd) else 'global']['codebook']
        self.codebook = torch.tensor(self.codebook).cuda(self.gpu_id)

    def update(self, update):
        self.prev_w = convert_np_to_tensor(update['model'], self.gpu_id)
        set_state_dict(self.model, update['model'], self.gpu_id, skip_stat=True, skip_mask=True)

    def on_round_begin(self):
        self.train()
        self.transfer_to_server()

    def train(self):
        st = time.time()
        val_local_acc, val_local_lss = self.eval(loader=self.loader.pa_loader_val)
        test_local_acc, test_local_lss = self.eval(loader=self.loader.pa_loader_test)
        self.logger.print(
            f'rnd:{self.curr_rnd + 1}, ep:{0}, '
            f'val_local_acc:{val_local_acc:.4f}, test_local_acc:{test_local_acc:.4f} ({time.time() - st:.2f}s)'
        )
        self.log['ep_local_val_acc'].append(val_local_acc)
        self.log['ep_local_val_lss'].append(val_local_lss)
        self.log['ep_local_test_acc'].append(test_local_acc)
        self.log['ep_local_test_lss'].append(test_local_lss)
        self.node_feature = []

        for ep in range(self.args.n_eps):
            st = time.time()
            self.model.train()
            for i, batch in enumerate(self.loader.pa_loader):
                self.optimizer.zero_grad()
                batch.codebook = self.codebook
                batch = batch.to(self.gpu_id)
                y_hat = self.model(batch)
                train_lss = F.nll_loss(y_hat, batch.y.view(-1))

                '''
                for name, param in self.model.state_dict().items():
                    if 'conv' in name or 'clsif' in name:
                        if self.curr_rnd > 0:
                            train_lss += torch.norm(param.float()-self.prev_w[name], 2) * self.args.loc_l2'''
                        
                train_lss.backward()
                self.optimizer.step()

                self.model.vq.update(self.model.node_feature.reshape(-1, self.args.hidden_dim))
                self.node_feature.append(self.model.node_feature.reshape(-1, self.args.hidden_dim))

            val_local_acc, val_local_lss = self.eval(loader=self.loader.pa_loader_val)
            test_local_acc, test_local_lss = self.eval(loader=self.loader.pa_loader_test)
            self.logger.print(
                f'rnd:{self.curr_rnd+1}, ep:{ep+1}, '
                f'val_local_acc:{val_local_acc:.4f}, test_local_acc:{test_local_acc:.4f} ({time.time()-st:.2f}s)'
            )
            self.log['train_lss'].append(train_lss.item())
            self.log['ep_local_val_acc'].append(val_local_acc)
            self.log['ep_local_val_lss'].append(val_local_lss)
            self.log['ep_local_test_acc'].append(test_local_acc)
            self.log['ep_local_test_lss'].append(test_local_lss)
        self.log['rnd_local_val_acc'].append(val_local_acc)
        self.log['rnd_local_val_lss'].append(val_local_lss)
        self.log['rnd_local_test_acc'].append(test_local_acc)
        self.log['rnd_local_test_lss'].append(test_local_lss)
        self.save_log()

    @torch.no_grad()
    def eval(self, loader):
        y_true = []
        y_pred = []
        loss_list = []
        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.gpu_id)
                pred = self.model(batch)
                loss_list.append(F.nll_loss(pred, batch.y.view(-1)).item())
                y_true.append(batch.y)
                y_pred.append(pred.argmax(1))

        y_pred = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        correct = (y_pred == y_true).sum()
        acc = correct.item() / len(y_true)

        return acc, np.mean(loss_list)

    def transfer_to_server(self):
        self.sd[self.client_id] = {
            'model': get_state_dict(self.model),
            'train_size': len(self.loader.partition),
            'codebook': self.model.vq.get_k().cpu()
        }




    
    
