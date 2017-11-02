import numpy as np
from sklearn.cluster import AgglomerativeClustering as Agglo
from sklearn.decomposition import KernelPCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances


class BinMerger:
    
    def __init__(self, embedding_by_column):
        self.embedding_by_column = embedding_by_column

    def _get_cols_and_pairwise_dist_btw_embeddings(self, variable):
    
        def get_begin_point_of_bin(col):
            return float(col[len(variable):].split('_(')[-1].split(',')[0])
        
        embedding_by_col = [(col, e) for col, e in self.embedding_by_column.items() if variable == col[:len(variable)]]
        embedding_by_col = sorted(embedding_by_col, key=lambda x: get_begin_point_of_bin(x[0]))

        cols = [x[0] for x in embedding_by_col]
        embeddings = np.array([x[1] for x in embedding_by_col])
        
        # De-noise embeddings
        if len(cols) > 2:
            kernel_pca = KernelPCA(n_components=2, kernel='cosine')
            dist_matrix = pairwise_distances(kernel_pca.fit_transform(embeddings), metric='cosine').astype(np.float64)
        else:
            kernel_pca = KernelPCA(n_components=1, kernel='cosine')
            dist_matrix = pairwise_distances(kernel_pca.fit_transform(embeddings), metric='cosine').astype(np.float64)
        return cols, dist_matrix

    def _clustering_embeddings(self, dist_matrix, return_score=False):
        
        def make_connectivity(num_bins):
            connectivity = np.eye(num_bins)
            for i in range(num_bins):
                if i < num_bins - 1:
                    connectivity[i][i+1] = 1
                if i >= 1:
                    connectivity[i][i-1] = 1
            return connectivity

        # Determine Optimal Number of Cluster
        scores = []
        conn = make_connectivity(len(dist_matrix))
        for n_cluster in range(2, len(dist_matrix)):
            agg = Agglo(n_cluster, affinity='precomputed', linkage='complete', connectivity=conn)
            cluster_label = agg.fit_predict(dist_matrix)
            scores.append(silhouette_score(dist_matrix, cluster_label, metric='precomputed'))

        # Clustering with Optimal Number of Cluster
        if len(scores) > 0:
            best_n = np.argmax(scores) + 2
            best_score = np.max(scores)

            agg = Agglo(best_n, affinity='precomputed', linkage='complete', connectivity=conn)
            cluster_label = agg.fit_predict(dist_matrix)
        else:
            best_score = 0
            cluster_label = [0, 1]
            
        if not return_score:
            return cluster_label
        if return_score:
            return cluster_label, best_score

    def _get_cols_by_cluster(self, cols, cluster_label):
        
        def get_begin_point_of_interval(x):
            return float(x.split('_')[-1].split(', ')[0].replace('(',''))
        
        cols_by_cluster = dict()

        cnt, prev_label = -1, -1
        for col, label in sorted(zip(cols, cluster_label), 
                                 key=lambda x:get_begin_point_of_interval(x[0])):
            if prev_label == label:
                cols_by_cluster[cnt].append(col)
            else:
                cnt += 1
                cols_by_cluster[cnt] = [col]
            prev_label = label
        
        return cols_by_cluster

    def _merge_bins(self, variable, return_score=False):
        
        def get_catogory_level_name(variable, col_name):
            return col_name[len(variable) + 1:]
        
        merged_bins = list()
        split_points = set()

        cols, dist_matrix = self._get_cols_and_pairwise_dist_btw_embeddings(variable)
        if return_score:
            cluster_label, score = self._clustering_embeddings(dist_matrix, return_score=True)
        else:
            cluster_label = self._clustering_embeddings(dist_matrix)
        cols_by_cluster = self._get_cols_by_cluster(cols, cluster_label)

        for cols in cols_by_cluster.values():
            intervals = [get_catogory_level_name(variable, x) for x in cols]
            begin = intervals[0].split(' ')[0]
            end = intervals[-1].split(' ')[1]
            merged_bins.append(' '.join([begin, end]))

            begin_point = float(begin.replace('(','').replace(',',''))
            end_point = float(end.replace(']','').replace(',',''))
            split_points.update([begin_point, end_point])
                
        split_points = sorted(split_points)
        
        if return_score:
            return merged_bins, score
        else:
            return merged_bins, split_points
    
    def get_merged_bins_by_var(self, var_dict):
        bins_by_variable = dict()
        for var in var_dict['numerical_vars']:
            merged_bins, split_points = self._merge_bins(var)
            bins_by_variable[var] = dict(bins=merged_bins, split_point=split_points)

        return bins_by_variable
