import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


def clustering(data, num_cluster, method):
    
    if type(data) not in [pd.DataFrame, np.ndarray]:
        raise ValueError('Type of data should be Pandas DataFrame or Numpy Array')
    
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=num_cluster)
        kmeans.fit(data)
        return kmeans.labels_
    
    elif method == 'agglomerative':
        agg = AgglomerativeClustering(n_clusters=num_cluster)
        agg.fit(data)
        return agg.labels_
        
    else:
        raise NotImplementedError

def eval_clustering_quality(cluster_label, class_label):
    score = normalized_mutual_info_score(cluster_label, class_label)
    # TODO: Add other measures for evaluating cluster quality
    return score

# def interpret_clusters(uniformly_binned_data, cluster_label):
#     # TODO: Implement Interpretation Method
#     # Output: Variable Names which are 