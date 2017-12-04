import numpy as np
from sklearn.cluster import AgglomerativeClustering as Agglo
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import pairwise_distances


class BinMerger:
    
    def __init__(self, embedding_by_column, co_occur_cutoff):
        self.embedding_by_column = embedding_by_column
        self.co_occur_cutoff = co_occur_cutoff

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

    def _clustering_embeddings(self, dist_matrix):

        def post_process_cluster_label(cluster_label):
            p_cluster_label = []
            cnt = 0
            prev_label = cluster_label[0]
            for l in cluster_label:
                if l == prev_label:
                    p_cluster_label.append(cnt)
                else:
                    cnt += 1
                    p_cluster_label.append(cnt)
                prev_label = l
            return p_cluster_label

        def ensemble_clustering(n_cols, results):
            adj_matrix = np.eye(n_cols) * self.co_occur_cutoff
            for i in range(n_cols):
                for result in results:
                    for j in range(n_cols):
                        if result[i] == result[j]:
                            adj_matrix[i][j] += 1

            adj_matrix = adj_matrix >= self.co_occur_cutoff # Cutoff for co-occurence value
            
            cluster_label = []
            i, c_label, cluster_start_idx = 0, 0, 0
            while(len(cluster_label) != len(adj_matrix)):
                for i in range(cluster_start_idx, len(adj_matrix)):
                    co_occur = adj_matrix[cluster_start_idx][i]
                    if co_occur == 0:
                        cluster_start_idx = i
                        c_label += 1
                        break
                    else:
                        cluster_label.append(c_label)
            return cluster_label

        if len(dist_matrix) == 2: # Do not merge
            return [0, 1]
        else:
            results = []
            for n_cluster in range(2, len(dist_matrix)):
                #agg = Agglo(n_cluster, affinity='precomputed', linkage='complete')
                kmeans = KMeans(n_cluster)
                #cluster_label = agg.fit_predict(dist_matrix)
                cluster_label = kmeans.fit_predict(dist_matrix)
                cluster_label = post_process_cluster_label(cluster_label)
                results.append(cluster_label)

            return ensemble_clustering(len(dist_matrix), results)

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
