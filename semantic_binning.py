import seaborn as sns
from data_handler import DataHandler
from embed_bins import BinEmbedder
from merge_bins import BinMerger


class SemanticBinning:
    
    def __init__(self, var_dict, embedding_dim, batch_size, n_epoch, lr, 
                 weight_decay=0.0, verbose=False, co_occur_cutoff=1):
        self.var_dict = var_dict
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.lr = lr
        self.weight_decay = weight_decay
        self.co_occur_cutoff = co_occur_cutoff
        self.verbose = verbose
        self.bin_embedder = BinEmbedder(co_occur_cutoff)
        
    def fit(self, data, n_init_bins=20):
        data_handler = DataHandler(data, self.var_dict)
        dummy_coded_data = data_handler.get_dummy_coded_data(n_init_bins=n_init_bins,)
                                                           # init_discretize_method='equal_width')
        
        self.bin_embedder.learn_bin_embeddings(dummy_coded_data,
                                               var_dict=self.var_dict,
                                               embedding_dim=self.embedding_dim,
                                               batch_size=self.batch_size,
                                               n_epoch=self.n_epoch,
                                               weight_decay=self.weight_decay,
                                               lr=self.lr,
                                               verbose=self.verbose)
        
        self._bin_merger = BinMerger(self.bin_embedder.embedding_by_column,
                                     co_occur_cutoff=self.co_occur_cutoff)
        self.bins_by_var = self._bin_merger.get_merged_bins_by_var(self.var_dict)

    def transform(self, data):
        data_handler = DataHandler(data, self.var_dict)
        return data_handler.get_dummy_coded_data(bins_by_variable=self.bins_by_var)
        
    def fit_transform(self, data, n_init_bins=20):
        self.fit(data, n_init_bins)
        return self.transform(data)
        
    def visualize_bin_embeddings(self, figsize=(20,20)):
        self.bin_embedder.visualize_embeddings(figsize)
    
    def plot_pairwise_distance_between_bins(self, variable):
        cols, dist_matrix = self._bin_merger._get_cols_and_pairwise_dist_btw_embeddings(variable)
        sns.heatmap(dist_matrix, cmap='coolwarm', yticklabels=cols)
