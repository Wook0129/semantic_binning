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
    
    def __init__(self, co_occur_cutoff=1):
        self._bins_by_var_reg = list()
        self.co_occur_cutoff = co_occur_cutoff

    def _generate_instances(self, dummy_coded_data, n_variables):
        inputs, targets = list(), list()
        for _, row in dummy_coded_data.iterrows():
            non_zero_idxs = [idx for idx, (_, value) in enumerate(row.items()) if value == 1]
            for target_idx in non_zero_idxs:
                inputs.append([idx for idx in non_zero_idxs if idx != target_idx])
                targets.append(target_idx)
        return inputs, targets
    
    def _get_current_cluster(self, dummy_coded_data, var_dict):
        embedding_weights = self.be.state_dict()['embedding.weight']
        embedding_by_column = dict(zip(list(dummy_coded_data.columns), embedding_weights.cpu().numpy()))
        bin_merger = BinMerger(embedding_by_column, self.co_occur_cutoff)
        num_bins = []
        bins_by_var = dict()
        for var in var_dict['numerical_vars']:
            merged_bins, split_points = bin_merger._merge_bins(var)
            num_bins.append(len(merged_bins))
            bins_by_var[var] = merged_bins
        return num_bins, bins_by_var
    
    def _check_convergence(self, bins_by_var, window_size=10):
        
        def is_equal_bin(bins_by_var1, bins_by_var2):
            cnt = 0
            for var, bins in bins_by_var1.items():
                is_equal = True
                for b1, b2 in zip(bins, bins_by_var2[var]):
                    if b1 != b2:
                        is_equal = False
                if is_equal:
                    cnt += 1
            if cnt == len(bins_by_var1):
                return True
            else:
                return False
            
        if len(self._bins_by_var_reg) < window_size:
            self._bins_by_var_reg.append(bins_by_var)
            return False
        else:
            convergence_cnt = 0
            for prev_bins_by_var in self._bins_by_var_reg[-window_size:]:
                if is_equal_bin(prev_bins_by_var, bins_by_var):
                    convergence_cnt += 1
            if convergence_cnt == window_size:
                return True
            else:
                self._bins_by_var_reg = self._bins_by_var_reg[1:] + [bins_by_var]
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
        
        torch.cuda.random.manual_seed_all(42)
        torch.manual_seed(42)
        
        self.be = BinEmbedding(n_dummy_cols, embedding_dim).cuda()

        loss_ftn = nn.CrossEntropyLoss()
        
        opt = torch.optim.Adagrad(self.be.parameters() , lr=lr, lr_decay=0.001)
        
        for it in range(n_iter_per_epoch * n_epoch):
            
            input_batch, target_batch = batch_gen.next_batch() 
            
            opt.zero_grad()
            
            input_batch = Variable(torch.LongTensor(input_batch)).cuda()
            target_batch = Variable(torch.LongTensor(target_batch)).cuda()
            
            out = self.be(input_batch)
            loss = loss_ftn(out, target_batch)
            
            loss.backward()
            opt.step()
            
            # Normalize Embedding Vectors
            embedding_norm = torch.norm(self.be.embedding.weight, p=2, dim=1).data
            embedding_norm = embedding_norm.view(-1,1).expand_as(self.be.embedding.weight)
            self.be.embedding.weight.data = self.be.embedding.weight.data.div(embedding_norm)

            if ((it+1) % n_iter_per_epoch == 0):
                
                if verbose:
                    print('>>> Epoch = {}'.format(int((it+1) / n_iter_per_epoch)))
                    print('Loss = {}'.format(loss.data[0]))
        
        embedding_weights = self.be.state_dict()['embedding.weight'].cpu().numpy()
        self.embedding_by_column = dict(zip(list(dummy_coded_data.columns), embedding_weights))
    
    def visualize_embeddings(self, figsize=(20,20), perplexity=10):
        
        col_names = list(self.embedding_by_column.keys())
        embedding_weights = list(self.embedding_by_column.values())
        tsne = TSNE(perplexity=perplexity).fit_transform(embedding_weights)
        
        plt.figure(figsize=figsize)
        plt.scatter(x=tsne[:,0], y=tsne[:,1])
        
        for i, xy in enumerate(zip(tsne[:,0], tsne[:,1])):
            plt.annotate(col_names[i], xy)
