import torch
from torch.nn import Linear as Lin, Sequential as Seq
import torch_geometric as tg
import torch.nn.functional as F
from torch import nn 
from PAE import PAE

class EV_GCN(torch.nn.Module):
    def __init__(self,  dropout, edgenet_input_dim, edge_dropout):
        super(EV_GCN, self).__init__()

        self.edge_dropout = edge_dropout 

        self.edge_net = PAE(input_dim=edgenet_input_dim//2, dropout=dropout)
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, edge_index, edgenet_input, enforce_edropout=False): 

        edge_weight = torch.squeeze(self.edge_net(edgenet_input))


        return edge_weight

