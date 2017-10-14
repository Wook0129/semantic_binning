from data_handler import DataHandler
from embed_bins import BinEmbedder
from merge_bins import BinMerger


class SemanticBinning:
    
    def __init__(self, var_dict, 
                 embedding_dim=8, n_epoch=20, lr=0.001, verbose=True,
                 clustering_method='agglomerative', merge_categorical_var=False):
        
        self.var_dict = var_dict
        
        self.embedding_dim = embedding_dim
        self.n_epoch = n_epoch
        self.lr = lr
        self.verbose = verbose
        
        self.clustering_method = clustering_method
        self.merge_categorical_var = merge_categorical_var
        
        self.bin_embedder = BinEmbedder()
        
    def fit(self, data, n_init_bins=20):
        
        data_handler = DataHandler(data, self.var_dict)
        dummy_coded_data = data_handler.get_dummy_coded_data(n_init_bins=n_init_bins)
        
        self.bin_embedder.learn_bin_embeddings(dummy_coded_data,
                                               data_handler.n_variables,
                                               embedding_dim=self.embedding_dim,
                                               n_epoch=self.n_epoch,
                                               lr=self.lr,
                                               verbose=self.verbose)
        
        bin_merger = BinMerger(self.bin_embedder.embedding_by_column, 
                               clustering_method=self.clustering_method)
        
        self.bins_by_var = bin_merger.get_merged_bins_by_var(self.var_dict,
                                                             merge_categorical_var=self.merge_categorical_var)

    def transform(self, data):
        data_handler = DataHandler(data, self.var_dict)
        return data_handler.get_dummy_coded_data(bins_by_variable=self.bins_by_var)
        
    def fit_transform(self, data, n_init_bins=20):
        self.fit(data, n_init_bins)
        return self.transform(data)
        
    def visualize_bin_embeddings(self, figsize=(20,20)):
        self.bin_embedder.visualize_embeddings(figsize)
 