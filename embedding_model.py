import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

   
class BinEmbedding(nn.Module):
      
    def __init__(self, n_dummy_columns, embedding_dim):
        super(BinEmbedding, self).__init__()
        self.embedding = nn.Embedding(n_dummy_columns, embedding_dim)
        self.decoder = nn.Linear(embedding_dim, n_dummy_columns, bias=False)
        
    def forward(self, x):
        h = self.embedding(x)
        out = torch.sum(self.decoder(h), dim=1)
        #out = self.decoder(h)
        return out
