import torch
import torch.nn as nn
import torch.nn.functional as F
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__( )
        self.w1 = nn.Linear(d_model,d_ff)
        self.w2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(0.1)
    def forward(self,embedings):
        embedings = self.w2(self.dropout(F.elu(self.w1(embedings))))
        return embedings