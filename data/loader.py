from misc.utils import *
from functools import partial

class Batch():
    def __init__(self, attn_bias, x, y, ids, centroids):
        super(Batch, self).__init__()
        self.x, self.y = x, y
        self.attn_bias = attn_bias
        self.ids = ids
        self.centroids = centroids

    def to(self, device):
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.attn_bias = self.attn_bias.to(device)
        self.ids = self.ids.to(device)
        if self.centroids is not None:
            self.centroids = self.centroids.to(device)
        return self

    def __len__(self):
        return self.y.size(0)

class DataLoader:
    def __init__(self, args, is_server=False):
        self.args = args
        self.n_workers = 1
        self.client_id = None

        if self.args.model == 'fedgt':
            from torch.utils.data import DataLoader
            self.DataLoader = DataLoader
        else:
            from torch_geometric.loader import DataLoader
            self.DataLoader = DataLoader
            
        if is_server and self.args.eval_global:
            self.test = get_data(self.args, mode='test')
            self.te_loader = self.DataLoader (dataset=self.test, batch_size=1, 
                shuffle=False, num_workers=self.n_workers, pin_memory=False)
            self.valid = get_data(self.args, mode='val')
            self.va_loader = self.DataLoader (dataset=self.valid, batch_size=1, 
                shuffle=False, num_workers=self.n_workers, pin_memory=False)

    def switch(self, client_id):
        if not self.client_id == client_id:
            self.client_id = client_id
            self.partition = get_data(self.args, client_id=client_id)
            self.pa_loader = self.DataLoader(dataset=self.partition, batch_size=1, 
                shuffle=False, num_workers=self.n_workers, pin_memory=False)
            
            if self.args.eval_global:
                self.test = get_data(self.args, mode='test', client_id=client_id)
                self.te_loader = self.DataLoader (dataset=self.test, batch_size=1, 
                    shuffle=False, num_workers=self.n_workers, pin_memory=False)
                self.valid = get_data(self.args, mode='val', client_id=client_id)
                self.va_loader = self.DataLoader (dataset=self.valid, batch_size=1, 
                    shuffle=False, num_workers=self.n_workers, pin_memory=False)

def get_data(args, client_id):
    return [
        torch_load(
            args.data_path, 
            f'{args.dataset}_{args.mode}/{args.n_clients}/partition_{client_id}.pt'
        )['client_data']
    ]
    
    
def collator(items, feature, labels, ppr, power_adj_list):
    items = torch.tensor(items)
    context = torch.multinomial(ppr[items], 25, replacement=False)
    y = labels[items]
    ids = torch.cat([items.unsqueeze(1), context], dim=1)
    x = feature[ids]
    attn_bias_list = []
    for b in range(ids.shape[0]):
        attn_bias_list.append(torch.cat([torch.tensor(m[ids[b], :][:, ids[b]].toarray(), dtype=torch.float32).unsqueeze(0) for m in power_adj_list]).permute(1, 2, 0).unsqueeze(0))
    attn_bias = torch.cat(attn_bias_list, dim=0) #(batch, id, id, bias_dim)
    return Batch(
        attn_bias=attn_bias,
        x=x,
        y=y,
        ids=ids,
        centroids=None,
    )
