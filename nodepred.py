import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv


class mlp(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(mlp, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = torch.nn.Linear(self.hidden_size, self.num_classes)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.relu(self.fc2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.relu(self.fc3(x))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.fc4(x)
        x = x.log_softmax(dim=-1)
        return x
    
    
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels) 
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        
    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr
        
    def forward(self, x, edge_index, edge_weight=None):
        #edge_index, _ = add_self_loops(edge_index)
        
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.lin(x)
        x = x.log_softmax(dim=-1)
        return x
    
    
class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(SAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels) 
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def set_aggr(self, aggr):
        self.conv1.aggr = aggr
        self.conv2.aggr = aggr
        self.conv3.aggr = aggr
        
    def forward(self, x, edge_index, edge_weight=None):
        #edge_index, _ = add_self_loops(edge_index)

        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.relu(self.conv2(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = F.relu(self.conv3(x, edge_index, edge_weight))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.lin(x)
        x = x.log_softmax(dim=-1)
        return x