import torch
import torch_geometric as pyg
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, TopKPooling
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

class GRAPH_CONV(nn.Module):
    def __init__(self, type, in_channels, out_channels):
        super(GRAPH_CONV, self).__init__()
        self.type = type
        self.in_channels = in_channels
        self.out_channels = out_channels
        if type == "gcn":
            self.graph_conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.graph_conv(x, edge_index)

class GRAPH_POOL(nn.Module):
    def __init__(self, type, in_channels, poolratio):
        super(GRAPH_POOL, self).__init__()
        self.type = type
        self.in_channels = in_channels
        self.poolratio = poolratio
        if self.type == "sagpool":
            self.graph_pool = SAGPooling(in_channels, ratio=poolratio)
        elif self.type == "topkpool":
            self.graph_pool = TopKPooling(in_channels, ratio=poolratio)
    
    def forward(self, x, edge_index, batch):
        return self.graph_pool(x, edge_index, batch=batch)

class GRAPH_READOUT(nn.Module):
    def __init__(self, type):
        super(GRAPH_READOUT, self).__init__()
        self.type = type
    
    def forward(self, x, batch):
        if self.type == "max":
            return global_max_pool(x, batch)
        elif self.type == "mean":
            return global_mean_pool(x, batch)
        elif self.type == "add":
            return global_add_pool(x, batch)

class GCN_Model(nn.Module):
    def __init__(self, inpDim) -> None:
        super(GCN_Model, self).__init__()
        self.layers = [GCNConv(inpDim, inpDim) for i in range(0, 5)]
        self.graph_readout = GRAPH_READOUT("max")

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x, edge_index)), p=0.2, training=True)
        out = self.graph_readout(x, None)
        return out

class Transformer_Model(nn.Module):
    def __init__(self, inpDim, nhead):
        super(Transformer_Model, self).__init__()
        encoderLayer = nn.TransformerEncoderLayer(inpDim, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoderLayer, 8)

    def forward(self, inp):
        inp = self.encoder(inp)
        return inp

class Pretrain_Model(nn.Module):
    pass

class HJ_Model(nn.Module):
    def __init__(self, ndim, nhead) -> None:
        super(HJ_Model, self).__init__()
        self.gcnLayer = GCN_Model(ndim)
        self.transformerLayer = Transformer_Model(ndim, nhead)
        self.finetuneLayer = nn.Sequential(
            nn.Linear(ndim, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
    
    def forward(self, inp):
        inp = self.gcnLayer(inp.x, inp.edge_index)
        inp = self.transformerLayer(inp)
        out = self.finetuneLayer(inp)
        return out
