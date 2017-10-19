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
        
        def make_target_vector(input_idx, non_zero_idxs, n_dummy_cols):
            target_vector = []
            for idx in range(n_dummy_cols):
                if (idx != input_idx and idx in non_zero_idxs):
                    target_vector.append(1.0)
                else:
                    target_vector.append(0.0)
            return target_vector
        
        inputs, targets = list(), list()
        n_dummy_cols = dummy_coded_data.shape[1]
        
        for _, row in dummy_coded_data.iterrows():
            non_zero_idxs = [idx for idx, (_, value) in enumerate(row.items()) if value == 1]
            for input_idx in non_zero_idxs:
                inputs.append(input_idx)
                targets.append(make_target_vector(input_idx, non_zero_idxs, n_dummy_cols))
                
        return inputs, targets

    def learn_bin_embeddings(self, dummy_coded_data, var_dict, embedding_dim,
                            lr, n_epoch, weight_decay, inter_bin_distance_penalty, batch_size, verbose):
        
        n_variables = len(var_dict['categorical_vars'] + var_dict['numerical_vars'])
        inputs, targets = self._generate_instances(dummy_coded_data, n_variables)
        
        n_instances = len(dummy_coded_data)
        n_dummy_cols = dummy_coded_data.shape[1]
        batch_size = min(int(n_instances / 10), batch_size)
        n_iter_per_epoch = int(np.ceil(n_instances * (n_variables - 1) / batch_size))
        batch_gen = BatchGenerator(inputs, targets, batch_size)
        
        dummy_cols = dummy_coded_data.columns
        
        self.bin_embedding = BinEmbedding(n_dummy_cols, embedding_dim).cuda()
        
        loss_ftn = nn.BCEWithLogitsLoss()
        
        opt = torch.optim.Adam(self.bin_embedding.parameters(), lr=lr, weight_decay=weight_decay)
        
        for it in range(n_iter_per_epoch * n_epoch):
            
            input_batch, target_batch = batch_gen.next_batch()
            
            opt.zero_grad()
            
            input_batch = Variable(torch.LongTensor(input_batch)).cuda()
            target_batch = Variable(torch.FloatTensor(target_batch)).cuda()
            
            out = self.bin_embedding(input_batch)

            ibd_loss = Variable(torch.FloatTensor([0.0]), requires_grad=True).cuda()
            for var in var_dict['numerical_vars']:
                col_idxs = [i for i, x in enumerate(dummy_cols) if var == x[:len(var)]]
                ibd_loss_input = Variable(torch.LongTensor(col_idxs)).cuda()
                embs = self.bin_embedding.embedding(ibd_loss_input)
                for i in range(len(embs) - 1):
                    ibd_loss += torch.norm(embs[i] - embs[i+1])
            
            loss = loss_ftn(out, target_batch) + ibd_loss * inter_bin_distance_penalty
            loss.backward()
            opt.step()

            if ((it+1) % n_iter_per_epoch == 0) and verbose:
                print('>>> Epoch = {}, Loss = {}'.format(int((it+1) / n_iter_per_epoch), loss.data[0]))
                
                ###
                from merge_bins import BinMerger
                embedding_weights = self.bin_embedding.state_dict()['embedding.weight']
                embedding_by_column = dict(zip(list(dummy_coded_data.columns), embedding_weights.cpu().numpy()))
                bin_merger = BinMerger(embedding_by_column)
                num_bins = []
                for var in var_dict['numerical_vars']:
                    merged_bins, _ = bin_merger._merge_bins(var)
                    num_bins.append(len(merged_bins))
                print(num_bins)
                ###
                
        if verbose:       
            print('Learning Embedding Finished!')
        
        embedding_weights = self.bin_embedding.state_dict()['embedding.weight'].cpu().numpy()
        self.embedding_by_column = dict(zip(list(dummy_coded_data.columns), embedding_weights))
    
    def visualize_embeddings(self, figsize=(20,20)):
        
        col_names = list(self.embedding_by_column.keys())
        embedding_weights = list(self.embedding_by_column.values())
        tsne = TSNE().fit_transform(embedding_weights)
        
        plt.figure(figsize=figsize)
        plt.scatter(x=tsne[:,0], y=tsne[:,1])
        
        for i, xy in enumerate(zip(tsne[:,0], tsne[:,1])):
            plt.annotate(col_names[i], xy)
