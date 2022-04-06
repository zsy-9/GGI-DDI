import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from Drug_Conv import ASAP_Pooling
from  interaction import Interactions
from torch_geometric.nn import GCNConv


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DD_Pre(torch.nn.Module):
    def __init__(self, in_channels, ratio1, ratio2):
        super(DD_Pre, self).__init__()
        self.in_channels = in_channels
        self.ratio1 = ratio1
        self.ratio2 = ratio2
        self.conv1 = GCNConv(self.in_channels, self.in_channels)
        self.pool1 = ASAP_Pooling(self.in_channels, self.ratio1)
        self.conv2 = GCNConv(self.in_channels, self.in_channels)
        self.pool2 = ASAP_Pooling(self.in_channels, self.ratio2)
        self.Interactions = Interactions(self.in_channels)#.to(device)
    def forward(self,Drug_1_E,Drug_1_x,D1_batch,Drug_2_E,Drug_2_x,D2_batch,Interaction):
        x_1 = F.relu(self.conv1(Drug_1_x, Drug_1_E))
        x_2 = F.relu(self.conv1(Drug_2_x, Drug_2_E))
        X_1_1, egde_1_1, batch_1_1,perm_1_1,fitness_1_1,index_S_1_1,value_S_1_1,edge_weight_1_1 = self.pool1(x_1, Drug_1_E, D1_batch)
        X_2_1, egde_2_1, batch_2_1,perm_2_1,fitness_2_1,index_S_2_1,value_S_2_1,edge_weight_2_1 = self.pool1(x_2, Drug_2_E, D2_batch)
        x_1 = F.relu(self.conv2(X_1_1, egde_1_1))
        x_2 = F.relu(self.conv2(X_2_1, egde_2_1))
        X_1_2, egde_1_2, batch_1_2, perm_1_2, fitness_1_2, index_S_1_2, value_S_1_2, edge_weight_1_2 = self.pool2(x_1, egde_1_1, batch_1_1)
        X_2_2, egde_2_2, batch_2_2, perm_2_2, fitness_2_2, index_S_2_2, value_S_2_2, edge_weight_2_2 = self.pool2(x_2, egde_2_1, batch_2_1)
        ans,ans_SSI = self.Interactions(X_1_2, batch_1_2, X_2_2, batch_2_2, Interaction)
        return ans#,ans_SSI

