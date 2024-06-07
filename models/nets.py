import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from misc.utils import *
from torch.autograd import Function, Variable

class VQ(nn.Module):
    def __init__(
            self,
            num_embeddings,
            embedding_dim,
            decay=0.99):
        super(VQ, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.register_buffer('vq_embedding', torch.rand(self._num_embeddings, self._embedding_dim)) #normalized codebook
        self.register_buffer('vq_embedding_output', torch.rand(self._num_embeddings, self._embedding_dim)) # output codebook
        self.register_buffer('vq_cluster_size', torch.ones(num_embeddings))

        self._decay = decay
        self.bn = torch.nn.BatchNorm1d(self._embedding_dim, affine=False, momentum=None)

    def get_k(self) :
        return self.vq_embedding_output

    def get_v(self) :
        return self.vq_embedding

    def update(self, x):
        inputs_normalized = self.bn(x)
        embedding_normalized = self.vq_embedding

        # Calculate distances
        distances = (torch.sum(inputs_normalized ** 2, dim=1, keepdim=True)
                     + torch.sum(embedding_normalized ** 2, dim=1)
                     - 2 * torch.matmul(inputs_normalized, embedding_normalized.t()))

        # FindNearest
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Use momentum to update the embedding vectors
        dw = torch.matmul(encodings.t(), inputs_normalized)
        self.vq_embedding.data = self.vq_cluster_size.unsqueeze(1) * self.vq_embedding * self._decay + (1 - self._decay) * dw
        self.vq_cluster_size.data = self.vq_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)

        self.vq_embedding.data = self.vq_embedding.data / self.vq_cluster_size.unsqueeze(1)

        # Output
        running_std = torch.sqrt(self.bn.running_var + 1e-5).unsqueeze(dim=0)
        running_mean = self.bn.running_mean.unsqueeze(dim=0)
        self.vq_embedding_output.data = self.vq_embedding*running_std + running_mean


class GCN(nn.Module):
    def __init__(self, n_feat=10, n_dims=128, n_clss=10, args=None):
        super().__init__()
        self.n_feat = n_feat
        self.n_dims = n_dims
        self.n_clss = n_clss
        self.args = args

        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(self.n_feat, self.n_dims, cached=False)
        self.conv2 = GCNConv(self.n_dims, self.n_dims, cached=False)
        self.clsif = nn.Linear(self.n_dims, self.n_clss)

    def forward(self, data, is_proxy=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        if is_proxy == True: return x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.clsif(x)
        return x

class MaskedGCN(nn.Module):
    def __init__(self, n_feat=10, n_dims=128, n_clss=10, l1=1e-3, args=None):
        super().__init__()
        self.n_feat = n_feat
        self.n_dims = n_dims
        self.n_clss = n_clss
        self.args = args
        
        from models.layers import MaskedGCNConv, MaskedLinear
        self.conv1 = MaskedGCNConv(self.n_feat, self.n_dims, cached=False, l1=l1, args=args)
        self.conv2 = MaskedGCNConv(self.n_dims, self.n_dims, cached=False, l1=l1, args=args)
        self.clsif = MaskedLinear(self.n_dims, self.n_clss, l1=l1, args=args)

    def forward(self, data, is_proxy=False):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        if is_proxy == True: return x
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.clsif(x)
        return x
    
    
def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size*2, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, attn_bias_dim):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_bias = nn.Linear(attn_bias_dim, num_heads)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        if attn_bias is not None:
            attn_bias = self.linear_bias(attn_bias).permute(0, 3, 1, 2)
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.num_heads * d_v)

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, num_heads, attn_bias_dim):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention_local = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads, attn_bias_dim)
        self.self_attention_dropout_local = nn.Dropout(dropout_rate)

        self.self_attention_global = MultiHeadAttention(
            hidden_size, attention_dropout_rate, num_heads, attn_bias_dim)
        self.self_attention_dropout_global = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size*2)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)


    def forward(self, x, c, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attention_local(y, y, y, attn_bias)
        y = self.self_attention_dropout_local(y)
        x = x + y

        g = self.self_attention_global(y, c, c, None)
        g = self.self_attention_dropout_global(g)
        g = x + g

        y = self.ffn_norm(torch.cat([x, g], dim=-1))
        #y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class GT(nn.Module):
    def __init__(
        self,
        n_layers,
        num_heads,
        input_dim,
        hidden_dim,
        output_dim,
        attn_bias_dim,
        dropout_rate,
        intput_dropout_rate,
        ffn_dim,
        num_global_node,
        attention_dropout_rate,
        num_centroids
    ):
        super().__init__()

        self.num_heads = num_heads
        self.node_encoder = nn.Linear(input_dim, hidden_dim)
        self.input_dropout = nn.Dropout(intput_dropout_rate)
        encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads, attn_bias_dim)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)
        self.n_layers = n_layers
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.downstream_out_proj = nn.Linear(hidden_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.num_global_node = num_global_node
        self.graph_token = nn.Embedding(self.num_global_node, hidden_dim)
        self.graph_token_virtual_distance = nn.Embedding(self.num_global_node, attn_bias_dim)
        self.node_feature = None
        self.vq = VQ(num_centroids, hidden_dim, decay=0.9)
        self.apply(lambda module: init_params(module, n_layers=n_layers))

    def forward(self, batched_data, perturb=None):
        attn_bias, x = batched_data.attn_bias, batched_data.x
        # graph_attn_bias
        n_graph, n_node = x.size()[:2]
        graph_attn_bias = attn_bias.clone()
        node_feature = self.node_encoder(x)
        self.node_feature = node_feature # [n_graph, n_node, n_hidden]
        if batched_data.centroids is None:
            c = self.vq.get_k()
            c = c.repeat(n_graph, 1, 1)
        else:
            c = batched_data.centroids.repeat(n_graph, 1, 1)
        if perturb is not None:
            node_feature += perturb

        global_node_feature = self.graph_token.weight.unsqueeze(0).repeat(n_graph, 1, 1)
        node_feature = torch.cat([node_feature, global_node_feature], dim=1)

        graph_attn_bias = torch.cat([graph_attn_bias, self.graph_token_virtual_distance.weight.unsqueeze(0).unsqueeze(2).repeat(n_graph, 1, n_node, 1)], dim=1)
        graph_attn_bias = torch.cat([graph_attn_bias, self.graph_token_virtual_distance.weight.unsqueeze(0).unsqueeze(0).repeat(n_graph, n_node+self.num_global_node, 1, 1)], dim=2)

        # transfomrer encoder
        output = self.input_dropout(node_feature)
        for enc_layer in self.layers:
            output = enc_layer(output, c, graph_attn_bias)
        output = self.final_ln(output)

        # output part
        output = self.downstream_out_proj(output[:, 0, :])
        return F.log_softmax(output, dim=1)



