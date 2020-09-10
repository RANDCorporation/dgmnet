import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SignedGCN, GCNConv, SAGEConv, ChebConv, MessagePassing, TopKPooling, global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (negative_sampling, remove_self_loops, add_self_loops)
from torch_geometric.utils import to_undirected
from sklearn.metrics import roc_auc_score, average_precision_score

EPS = 1e-10
class Net(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, embed_dim=64):
        super(Net, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        in_channels = num_node_features
        out_channels = embed_dim
        self.decoder = InnerProductDecoder().to(self.device)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        #self.lin = torch.nn.Linear(3 * hidden_channels, out_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)        
        self.linear_decoder1 = torch.nn.Linear(1, 1)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr

    def forward(self, x0, edge_index, edge_weight=None):
        x1 = F.tanh(self.conv1(x0, edge_index, edge_weight))
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x2 = F.tanh(self.conv2(x1, edge_index, edge_weight))
        x2 = F.dropout(x2, p=0.2, training=self.training)
        x3 = F.tanh(self.conv3(x2, edge_index, edge_weight))
        x3 = F.dropout(x3, p=0.2, training=self.training)
        #x = torch.cat([x1, x2, x3], dim=-1)
        #x = self.lin(x)
        x = self.lin(x3)
        return x
    
    def pij(self, z, edge_index):
        #edge_index, _ = remove_self_loops(edge_index)
        zizj = torch.transpose(self.decoder(z, edge_index, sigmoid=False)[None,:], 0, 1)
        zizj = F.sigmoid(self.linear_decoder1(zizj))[:,0]
        return zizj
        #return self.decoder(z, edge_index, sigmoid=True)
    
    
    def recon_loss(self, z, pos_edge_index):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """
        #pos_loss = -torch.log(
        #   self.decoder(z, pos_edge_index, sigmoid=True) + EPS).sum()
        pos_loss = -torch.log(self.pij(z, pos_edge_index) + EPS).sum()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)

        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        #neg_loss = -torch.log(1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS).sum()
        neg_loss = -torch.log(1 - self.pij(z, neg_edge_index) + EPS).sum()

        return pos_loss + neg_loss

    def test(self, z, pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        #pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        #neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pos_pred = self.pij(z, pos_edge_index)
        neg_pred = self.pij(z, neg_edge_index)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu(), pred.detach().cpu()

        return y, pred
        #return roc_auc_score(y, pred), average_precision_score(y, pred)


class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""

    def forward(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


    def forward_all(self, z, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj
