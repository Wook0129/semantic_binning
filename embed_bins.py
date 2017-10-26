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
    
    def __init__(self):
        self._scores_reg = list()
        
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
        scores = []
        for var in var_dict['numerical_vars']:
            merged_bins, score = bin_merger._merge_bins(var, return_score=True)
            num_bins.append(len(merged_bins))
            scores.append(score)
        return num_bins, scores
    
    def _check_convergence(self, curr_score, window_size=5):
        
        if len(self._scores_reg) < window_size:
            self._scores_reg.append(curr_score)
            return False
        else:
            convergence_cnt = 0
            for prev_score in self._scores_reg[-window_size:]:
                if np.abs(curr_score - prev_score) < 1e-2:
                    convergence_cnt += 1
            if convergence_cnt >= 3:
                return True
            else:
                self._scores_reg = self._scores_reg[1:] + [curr_score]
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
        
        self.bin_embedding = BinEmbedding(n_dummy_cols, embedding_dim).cuda()
        
        loss_ftn = nn.BCEWithLogitsLoss()
        
        opt = torch.optim.Adam(self.bin_embedding.parameters(), lr=lr, weight_decay=weight_decay)
        
        for it in range(n_iter_per_epoch * n_epoch):
            
            input_batch, target_batch = batch_gen.next_batch()
            
            opt.zero_grad()
            
            input_batch = Variable(torch.LongTensor(input_batch)).cuda()
            target_batch = Variable(torch.FloatTensor(target_batch)).cuda()
            
            out = self.bin_embedding(input_batch)
            loss = loss_ftn(out, target_batch)
            
            loss.backward()
            opt.step()

            if ((it+1) % n_iter_per_epoch == 0):

                num_bins, scores = self._get_current_cluster(dummy_coded_data, var_dict)
                curr_score = np.mean(scores)
                
                if verbose:
                    print('>>> Epoch = {}, Loss = {}'.format(int((it+1) / n_iter_per_epoch), loss.data[0]))
                    print(num_bins, curr_score)
                    
                if self._check_convergence(curr_score):
                    if verbose:
                        print('Embedding Converged!')
                    break
                    
        if not self._check_convergence(curr_score):        
            print('Embedding Failed to Converge..')

        num_bins, scores = self._get_current_cluster(dummy_coded_data, var_dict)    
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
