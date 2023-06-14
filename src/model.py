import torch
import torch_geometric as pyg
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGPooling, TopKPooling

class GCNLayer(nn.Module):
    def __init__(self, 
                 idim: int, 
                 hdim: int = 200, 
                 odim: int = 200, 
                 olen: int = 512) -> None:
        super(GCNLayer, self).__init__()
        self.layers = [GCNConv(idim, hdim),
                       GCNConv(hdim, hdim),
                       GCNConv(hdim, odim)]
        self.pooling = SAGPooling(200, 0.5)

    def forward(self, data: pyg.data.Data):
        x = data.x
        edge_index = data.edge_index
        for layer in self.layers:
            x = F.dropout(F.relu(layer(x, edge_index)), p=0.2, training=True)
        while True:
            [x, edge_index, _, _, _, _] = self.pooling(x, edge_index)
            if x.shape[0] <= 512:
                break
        pad = (0, 0, 0, 512 - x.shape[0])
        out = F.pad(x, pad, 'constant', 0.)
        return out

class TransformerLayer(nn.Module):
    def __init__(self, idim, nhead):
        super(TransformerLayer, self).__init__()
        encoderLayer = nn.TransformerEncoderLayer(idim, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoderLayer, 8)

    def forward(self, inp):
        inp = self.encoder(inp)
        return inp

class HJ_Model(nn.Module):
    def __init__(self, 
                 idim: int, 
                 nhead: int = 2) -> None:
        super(HJ_Model, self).__init__()
        self.gcnlayer = GCNLayer(idim)
        self.transformerlayer = TransformerLayer(200, nhead)
        self.outputlayer = nn.Sequential(
            nn.Linear(200 * 512, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )
    
    def forward(self, inp):
        inp = self.gcnlayer(inp)
        inp = self.transformerlayer(inp)
        inp = torch.reshape(inp, (1, 200 * 512))
        out = self.outputlayer(inp)
        return out

class HJ_LocModel(nn.Module):
    def __init__(self, 
                 idim: int, 
                 nhead: int = 2) -> None:
        super(HJ_LocModel, self).__init__()
        self.gcnlayer = GCNLayer(idim)
        self.transformerlayer = TransformerLayer(200, nhead)
        self.outputlayer = nn.Sequential(
            nn.Linear(idim, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
    
    def forward(self, inp):
        inp = self.gcnlayer(inp.x, inp.edge_index)
        inp = self.transformerlayer(inp)
        out = self.outputlayer(inp)
        return out