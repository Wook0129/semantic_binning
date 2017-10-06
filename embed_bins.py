import torch
from torch import nn
from torch.autograd import Variable

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from batch_generator import BatchGenerator
from embedding_model import BinEmbedding


class BinEmbedder:    
    
    def _generate_instances(self, dummy_coded_data, n_variables):
        
        n_dummy_columns = dummy_coded_data.shape[1]
        inputs, targets = list(), list()
        
        for _, row in dummy_coded_data.iterrows():
            
            non_zero_idxs = [idx for idx, (_, value) in 
                                 enumerate(row.items()) if value == 1]
            
            for input_idx in non_zero_idxs:
                inputs.append(input_idx)
                target_idxs = [idx for idx in non_zero_idxs if idx != input_idx]
                targets += [[1.0 if idx in target_idxs else 0.0 
                                for idx in range(0, n_dummy_columns)]]

        return inputs, targets

    def learn_bin_embeddings(self, dummy_coded_data, n_variables, embedding_dim=16,
                            lr=0.001, max_iter=300000, batch_size=256, verbose=True):

        inputs, targets = self._generate_instances(dummy_coded_data, n_variables)
        batch_gen = BatchGenerator(inputs, targets, batch_size)
        
        self.bin_embedding = BinEmbedding(dummy_coded_data.shape[1], embedding_dim).cuda()
        loss_ftn = nn.BCELoss()
        opt = torch.optim.Adam(self.bin_embedding.parameters(), lr=lr)
        
        for i in range(max_iter):
            
            input_batch, target_batch = batch_gen.next_batch()
            
            opt.zero_grad()
            input_batch = Variable(torch.LongTensor(input_batch)).cuda()
            target_batch = Variable(torch.FloatTensor(target_batch)).cuda()
            out = self.bin_embedding(input_batch)
            loss = loss_ftn(out, target_batch)
            loss.backward()
            opt.step()
            
            if ((i+1) % 10000 == 0) and verbose:
                print('>>> Iteration = {}, Loss = {}'.format((i+1), loss.data[0]))
                
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


#     def _generate_instances(self, dummy_coded_data, n_variables):
#         inputs = []
#         targets = []
#         for _, row in dummy_coded_data.iterrows():
#             non_zero_idxs = [idx for idx, (_, value) in enumerate(row.items()) if value == 1]
#             for input_idx in non_zero_idxs:
#                 inputs += [input_idx] * (n_variables - 1)
#                 targets += [idx for idx in non_zero_idxs if idx != input_idx]
#         return inputs, targets