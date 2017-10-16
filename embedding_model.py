import torch
from torch import nn
from torch.autograd import Variable


#class BinEmbedding(nn.Module):
    
#    def __init__(self, n_dummy_columns, embedding_dim):
#        super(BinEmbedding, self).__init__()
#        self.embedding = nn.Embedding(n_dummy_columns, embedding_dim)
#        self.decoder = nn.Linear(embedding_dim, n_dummy_columns, bias=True)
    
#    def forward(self, x):
#        h = self.embedding(x)
#        out = self.decoder(h)
#        return out

    
class BinEmbedding(nn.Module):
      
    def __init__(self, n_dummy_columns, embedding_dim):
        super(BinEmbedding, self).__init__()
        self.embedding = nn.Embedding(n_dummy_columns, embedding_dim)
        self.decoder = nn.Linear(embedding_dim, n_dummy_columns, bias=True)
        #self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h = self.embedding(x)
        #h = torch.sum(self.embedding(x), dim=1)
        out = self.decoder(h)
        #out = self.sigmoid(self.decoder(h))
        return out
