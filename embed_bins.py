import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from batch_generator import BatchGenerator
from embedding_model import BinEmbedding
from merge_bins import BinMerger


class BinEmbedder:
    
    def __init__(self, tol_num=5):
        self._num_bins_reg = list()
        self._tol_num = tol_num
        
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

    def _get_current_cluster(self, dummy_coded_data, var_dict):
        embedding_weights = self.bin_embedding.state_dict()['embedding.weight']
        embedding_by_column = dict(zip(list(dummy_coded_data.columns), embedding_weights.cpu().numpy()))
        bin_merger = BinMerger(embedding_by_column)
        num_bins = []
        for var in var_dict['numerical_vars']:
            merged_bins, _ = bin_merger._merge_bins(var)
            num_bins.append(len(merged_bins))
        return num_bins
    
    def _check_convergence(self, num_bins):
        if len(self._num_bins_reg) < self._tol_num:
            self._num_bins_reg.append(num_bins)
            return False
        else:
            converge_cnt = 0
            for n_bins in self._num_bins_reg:
                if np.mean(np.equal(num_bins, n_bins)) == 1:
                    converge_cnt += 1
            print(converge_cnt, self._tol_num)
            if converge_cnt == self._tol_num:
                return True
            else:
                self._num_bins_reg = self._num_bins_reg[1:] + [num_bins]
                return False
            
    def learn_bin_embeddings(self, dummy_coded_data, var_dict, embedding_dim,
                            lr, n_epoch, weight_decay, batch_size, verbose):
        
        n_variables = len(var_dict['numerical_vars'])
        if 'categorical_vars' in var_dict:
            n_variables += len(var_dict['categorical_vars'])
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
        
        num_bin_reg = []
        
        for it in range(n_iter_per_epoch * n_epoch):
            
            input_batch, target_batch = batch_gen.next_batch()
            
            opt.zero_grad()
            
            input_batch = Variable(torch.LongTensor(input_batch)).cuda()
            target_batch = Variable(torch.FloatTensor(target_batch)).cuda()
            
            out = self.bin_embedding(input_batch)
            loss = loss_ftn(out, target_batch)

            for var in var_dict['numerical_vars']:
                col_idxs = [i for i, x in enumerate(dummy_cols) if var == x[:len(var)]]
                embs = self.bin_embedding.embedding(Variable(torch.LongTensor(col_idxs)).cuda())
                for i in range(len(embs) - 1):
                    loss += 1e-7 * torch.norm(embs[i] - embs[i+1]) / (2 * len(embs))

            loss.backward()
            opt.step()

            if ((it+1) % n_iter_per_epoch == 0):

                num_bins = self._get_current_cluster(dummy_coded_data, var_dict)
                
                if verbose:
                    print('>>> Epoch = {}, Loss = {}'.format(int((it+1) / n_iter_per_epoch), loss.data[0]))
                    print(num_bins)
                    
                if self._check_convergence(num_bins):
                    if verbose:
                        print('Embedding Converged!')
                    break
                    
        if not self._check_convergence(num_bins):        
            print('Embedding Failed to Converge in given #epochs..')
            
        print('Learned #Bin by Variables = {}'.format(num_bins))
        
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
