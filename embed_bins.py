import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from batch_generator import BatchGenerator
from embedding_model import BinEmbedding


class BinEmbedder:    
    
    def _generate_instances(self, dummy_coded_data, n_variables):
        
        inputs, targets = list(), list()
        n_dummy_cols = dummy_coded_data.shape[1]
        
        for _, row in dummy_coded_data.iterrows():
            non_zero_idxs = [idx for idx, (_, value) in enumerate(row.items()) if value == 1]
            for input_idx in non_zero_idxs:
                inputs.append(input_idx)
                targets.append([1.0 if (idx != input_idx and idx in non_zero_idxs) else 0.0 for idx in range(n_dummy_cols)])
        return inputs, targets    

#    def _generate_instances(self, dummy_coded_data, n_variables):
        
#        inputs, targets = list(), list()
        
#        for _, row in dummy_coded_data.iterrows():
#            non_zero_idxs = [idx for idx, (_, value) in enumerate(row.items()) if value == 1]
#            for target_idx in non_zero_idxs:
#                inputs.append([idx for idx in non_zero_idxs if idx != target_idx])
#                targets.append(target_idx)
#        return inputs, targets    
    
    def learn_bin_embeddings(self, dummy_coded_data, n_variables, embedding_dim,
                            lr, n_epoch, weight_decay, batch_size, verbose):
        
        inputs, targets = self._generate_instances(dummy_coded_data, n_variables)
        
        n_instances = len(dummy_coded_data)
        batch_size = min(int(n_instances / 10), batch_size)
        n_iter_per_epoch = int(np.ceil(n_instances * (n_variables - 1) / batch_size))
        batch_gen = BatchGenerator(inputs, targets, batch_size)
        
        self.bin_embedding = BinEmbedding(dummy_coded_data.shape[1], embedding_dim).cuda()

        #loss_ftn = nn.CrossEntropyLoss()
        #loss_ftn = nn.BCELoss()
        loss_ftn = nn.BCEWithLogitsLoss()
        opt = torch.optim.Adam(self.bin_embedding.parameters(), lr=lr, weight_decay=weight_decay)
        
        for i in range(n_iter_per_epoch * n_epoch):
            
            input_batch, target_batch = batch_gen.next_batch()
            
            opt.zero_grad()
            input_batch = Variable(torch.LongTensor(input_batch)).cuda()
            target_batch = Variable(torch.FloatTensor(target_batch)).cuda()
            out = self.bin_embedding(input_batch)
            loss = loss_ftn(out, target_batch)
            loss.backward()
            opt.step()
            
            if ((i+1) % n_iter_per_epoch == 0) and verbose:
                print('>>> Epoch = {}, Loss = {}'.format(int((i+1) / n_iter_per_epoch), loss.data[0]))
                
        if verbose:       
            print('Learning Embedding Finished!')
        
        embedding_weights = self.bin_embedding.state_dict()['embedding.weight'].cpu().numpy()
        self.embedding_by_column = dict(zip(list(dummy_coded_data.columns), embedding_weights))
    
    def visualize_embeddings(self, figsize=(20,20)):
        
        col_names = list(self.embedding_by_column.keys())
        embedding_weights = list(self.embedding_by_column.values())
        
        tsne = TSNE()
        embedding_weights_tsne = tsne.fit_transform(embedding_weights)
        
        plt.figure(figsize=figsize)
        plt.scatter(x=embedding_weights_tsne[:,0], y=embedding_weights_tsne[:,1])
        
        for i, xy in enumerate(zip(embedding_weights_tsne[:,0], embedding_weights_tsne[:,1])):
            plt.annotate(col_names[i], xy)
