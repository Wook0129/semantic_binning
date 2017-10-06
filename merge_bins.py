import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class BinMerger:
    
    def __init__(self, embedding_by_column):
        self.embedding_by_column = embedding_by_column
    
    def _get_cols_and_embeddings(self, variable):
        cols = []
        embeddings = []
        for c, e in self.embedding_by_column.items():
            if variable in c:
                cols.append(c)
                embeddings.append(e)
        return cols, embeddings

    def _cluster_embeddings(self, embeddings):
        
        n_embeddings = len(embeddings)
        
        # Do not Merge Bins, if #Bins <= 2
        if n_embeddings <= 2:
            return [x for x in range(0, n_embeddings)]
        
        # Determine Optimal Number of Cluster
        scores = []
        for n_cluster in range(2, n_embeddings):
            cluster_label = KMeans(n_cluster).fit_predict(embeddings)
            score = silhouette_score(embeddings, cluster_label)
            scores.append(score)
        
        # Clustering with Optimal Number of Cluster
        best_n = np.argmax(scores) + 2
        cluster_label = KMeans(best_n).fit_predict(embeddings)
        
        return cluster_label

    def _get_cols_by_cluster(self, cols, cluster_label, v_type):
        
        cols_by_cluster = dict()

        if v_type == 'numerical':
            cnt, prev_label = -1, -1
            for col, label in sorted(zip(cols, cluster_label), key=lambda x:x[0]):
                if prev_label == label:
                    cols_by_cluster[cnt].append(col)
                else:
                    cnt += 1
                    cols_by_cluster[cnt] = [col]
                prev_label = label
                
        elif v_type == 'categorical':
            for col, label in zip(cols, cluster_label):
                if label in cols_by_cluster:
                    cols_by_cluster[label].append(col)
                else:
                    cols_by_cluster[label] = [col]
        
        else:
            raise ValueError('Available v_type = [numerical, categorical]')

        return cols_by_cluster

    def _merge_bins(self, variable, v_type='numerical'):
        
        merged_bins = list()
        split_points = set()
        
        cols, embeddings = self._get_cols_and_embeddings(variable)
        cluster_label = self._cluster_embeddings(embeddings)
        cols_by_cluster = self._get_cols_by_cluster(cols, cluster_label, v_type)
        
        for cols in cols_by_cluster.values():
        
            if len(cols) > 1:

                if v_type == 'numerical':
                    intervals = [x.split('_')[-1] for x in cols]
                    begin = intervals[0].split(' ')[0]
                    end = intervals[-1].split(' ')[1]
                    merged_bins.append(' '.join([begin, end]))
                    
                    begin_point = float(begin.replace('(','').replace(',',''))
                    end_point = float(end.replace(']','').replace(',',''))
                    split_points.update([begin_point, end_point])
                    
                if v_type == 'categorical':
                    category_levels = [x.split('_')[-1] for x in cols]
                    merged_bins.append(', '.join(category_levels))
                    
        split_points = sorted(split_points)
        
        return merged_bins, split_points
    
    def get_merged_bins_by_var(self, var_dict):
        
        bins_by_variable = dict()
        
        for var in var_dict['numerical_vars']:
            merged_bins, split_points = self._merge_bins(var, v_type='numerical')
            bins_by_variable[var] = dict(merged_bins=merged_bins, split_point=split_points)

        for var in var_dict['categorical_vars']:
            merged_bins, _ = self._merge_bins(var, v_type='categorical')
            bins_by_variable[var] = dict(merged_bins=merged_bins)
            
        return bins_by_variable
