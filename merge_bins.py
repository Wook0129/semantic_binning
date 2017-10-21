import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances


class BinMerger:
    
    def __init__(self, embedding_by_column):
        self.embedding_by_column = embedding_by_column

    def _get_cols_and_pairwise_dist_btw_embeddings(self, variable):
    
        def get_begin_point_of_bin(col):
            return float(col.split('_(')[-1].split(',')[0])

        embedding_by_col = [(col, e) for col, e in self.embedding_by_column.items() if variable in col]
        embedding_by_col = sorted(embedding_by_col, key=lambda x: get_begin_point_of_bin(x[0]))

        cols = [x[0] for x in embedding_by_col]
        dist_matrix = pairwise_distances(np.array([x[1] for x in embedding_by_col]), metric='cosine').astype(np.float64)

        return cols, dist_matrix

    def _clustering_embeddings(self, dist_matrix):

        # Determine Optimal Number of Cluster
        scores = []
        for n_cluster in range(2, len(dist_matrix)):
            cluster_label = KMeans(n_cluster).fit_predict(dist_matrix)
            try:
                scores.append(silhouette_score(dist_matrix, cluster_label))
            except:
                scores.append(0)

        # Clustering with Optimal Number of Cluster
        best_n = np.argmax(scores) + 2
        cluster_label = KMeans(best_n).fit_predict(dist_matrix)

        return cluster_label

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

    def _merge_bins(self, variable):
        
        def get_catogory_level_name(variable, col_name):
            return col_name[len(variable) + 1:]
        
        merged_bins = list()
        split_points = set()

        cols, dist_matrix = self._get_cols_and_pairwise_dist_btw_embeddings(variable)
        
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
        
        return merged_bins, split_points
    
    def get_merged_bins_by_var(self, var_dict):
        bins_by_variable = dict()
        for var in var_dict['numerical_vars']:
            merged_bins, split_points = self._merge_bins(var)
            bins_by_variable[var] = dict(bins=merged_bins, split_point=split_points)

        return bins_by_variable
