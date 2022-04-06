import torch
from torch import nn
from torch_geometric.nn.inits import glorot
from torch_geometric.utils import degree
from torch_geometric.utils import softmax, degree
from torch_scatter import scatter

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Interactions(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.bias = nn.Parameter(torch.zeros(in_channels))
        self.tr_inter = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.in_channels, self.in_channels),
            nn.PReLU(),
            nn.Linear(self.in_channels, self.in_channels)
        )
        self.mlp = nn.Sequential(
            nn.PReLU(),
            nn.Linear(self.in_channels*3, self.in_channels),
            nn.PReLU(),
            nn.Linear(self.in_channels, self.in_channels),
        )
        self.weight_1 = nn.Parameter(torch.zeros(in_channels, in_channels))
        self.weight_2 = nn.Parameter(torch.zeros(in_channels, in_channels))
        self.a = nn.Parameter(torch.zeros(in_channels))
        glorot(self.weight_1)
        glorot(self.weight_2)
        glorot(self.a.view(1, -1))
    def forward(self, x_1, batch_1, x_2, batch_2, inter):
        x_1 = x_1 @ self.weight_1
        x_2 = x_2 @ self.weight_2
        d_1 = degree(batch_1, dtype=batch_1.dtype)
        d_2 = degree(batch_2, dtype=batch_2.dtype)
        s_1 = torch.cumsum(d_1, dim=0)
        s_2 = torch.cumsum(d_2, dim=0)
        ind_1 = torch.cat([torch.arange(i,device = device).repeat_interleave(j) + (s_1[e - 1] if e else 0) for e, (i, j) in enumerate(zip(d_1, d_2))])
        ind_2 = torch.cat([torch.arange(j,device = device).repeat(i) + (s_2[e - 1] if e else 0) for e, (i, j) in enumerate(zip(d_1, d_2))])
        x_1 = x_1[ind_1]
        x_2 = x_2[ind_2]
        rels = self.tr_inter(inter)
        size_1_2=torch.mul(d_1,d_2)
        rels = torch.repeat_interleave(rels,size_1_2,dim=0)
        inputs = torch.cat((rels, x_1, x_2), 1)
        ans_SSI = (self.a * self.mlp(inputs)).sum(-1)
        batch_ans = torch.arange(inter.shape[0],device = device).repeat_interleave(size_1_2, dim=0)
        ans = scatter(ans_SSI,batch_ans,reduce='add',dim=0)
        return ans,ans_SSI










