import numpy as np
import pandas as pd
from clustering import clustering
from uniform_binning import UniformBinning


class SemanticBinning:
    
    def __init__(self, *, uniform_bin_numbers=20):
        self.uniform_bin_numbers = uniform_bin_numbers
        # Num of Bin Clusters for Each Variable Should be Adjustable by Init Method
        
    def binning(self, data, numerical_variable_names):
        
        uniform_binner = UniformBinning(bin_numbers=self.uniform_bin_numbers)
        uniformly_binned_data = uniform_binner.binning(data, numerical_variable_names)

        semantically_binned_data = uniformly_binned_data.copy()

        for variable_name in numerical_variable_names:
            bin_vector_dict = self._make_bin_vector_dict(variable_name, uniformly_binned_data)
            bin_labels_by_cluster = self._clustering_bins(bin_vector_dict)

            for cluster_label in bin_labels_by_cluster.keys():

                bin_labels = bin_labels_by_cluster[cluster_label]
                bin_boundaries = self._bin_labels_to_bin_boundaries(bin_labels)

                consolidated_bin_boundaries = self._consolidate_bin_boundaries(bin_boundaries)
                consolidated_bin_label = self._consolidate_bin_labels(variable_name, consolidated_bin_boundaries)

                consolidated_bin = uniformly_binned_data[bin_labels].values.sum(axis=1)
                semantically_binned_data[consolidated_bin_label] = consolidated_bin

                if len(bin_labels) > 1:
                    for bin_label in bin_labels:
                        semantically_binned_data.drop(bin_label, axis=1, inplace=True)

        return semantically_binned_data

    def _make_bin_vector_dict(self, variable_name, uniformly_binned_data):

        def bin_to_vector(data, bin_label, other_bin_labels):
            bin_related_data = data.loc[data[bin_label] == 1, other_bin_labels]
            vector_representation_of_bin = np.array(bin_related_data.mean(axis=0))
            return vector_representation_of_bin

        bin_vector_dict = dict()
        bin_labels = [col for col in uniformly_binned_data.columns if variable_name == '_'.join(col.split('_')[:-1])]
        other_bin_labels = [col for col in uniformly_binned_data.columns if variable_name != '_'.join(col.split('_')[:-1])]

        for bin_label in bin_labels:
            bin_vector_dict[bin_label] = bin_to_vector(uniformly_binned_data, 
                                                       bin_label, other_bin_labels)
        return bin_vector_dict

    def _clustering_bins(self, bin_vector_dict):

        bin_labels_by_cluster = dict()

        bin_labels = [bin_label for bin_label in bin_vector_dict.keys()]
        bin_vectors = np.array([vector for vector in bin_vector_dict.values()])
        bin_cluster_labels = clustering(bin_vectors, num_cluster=5, method='kmeans')
        # TODO: (1)Use KL-Divergence Metric and (2)Automatically Find Number of Clusters
        # It may require additional codes for implementing clustering method
        for cluster_label, bin_label in zip(bin_cluster_labels, bin_labels):
            if cluster_label in bin_labels_by_cluster:
                bin_labels_by_cluster[cluster_label].append(bin_label)
            else:
                bin_labels_by_cluster[cluster_label] = [bin_label]

        return bin_labels_by_cluster

    def _bin_labels_to_bin_boundaries(self, bin_labels):

        bin_boundaries = []

        for bin_label in bin_labels:

            bin_boundary = bin_label.split('_')[-1]

            if bin_boundary[0] == '[' and bin_boundary[-1] == ')':
                bin_boundary = bin_boundary[1:-1].split(', ')
            elif bin_boundary[0] == '[':
                bin_boundary = [bin_boundary[1:], None]
            elif bin_boundary[-1] == ')':
                bin_boundary = [None, bin_boundary[:-1]]

            bin_boundaries.append(bin_boundary)

        return bin_boundaries

    def _consolidate_bin_boundaries(self, bin_boundaries):

        def possible_to_consolidate(bin_boundaries):
            for i, bin_boundary1 in enumerate(bin_boundaries):
                for j, bin_boundary2 in enumerate(bin_boundaries[i:]):
                    if (bin_boundary1[1] == bin_boundary2[0]) or (bin_boundary1[0] == bin_boundary2[1]):
                        return True
            return False

        while(possible_to_consolidate(bin_boundaries)):
            for i, x in enumerate(bin_boundaries):
                for j, y in enumerate(bin_boundaries[i:]):
                    if (x[1] == y[0]):
                        bin_boundaries.remove(x)
                        bin_boundaries.remove(y)
                        bin_boundaries.append([x[0], y[1]])
                        break
                    elif (x[0] == y[1]):
                        bin_boundaries.remove(x)
                        bin_boundaries.remove(y)
                        bin_boundaries.append([y[0], x[1]])
                        break
                    else:
                        pass

        return bin_boundaries

    def _consolidate_bin_labels(self, variable_name, consolidated_bin_boundaries):

        bin_labels = []

        for bin_boundary in consolidated_bin_boundaries:
            if bin_boundary[0] == None:
                bin_labels.append('{}_{})'.format(variable_name, bin_boundary[1]))
            elif bin_boundary[1] == None:
                bin_labels.append('{}_[{}'.format(variable_name, bin_boundary[0]))
            else:
                bin_labels.append('{}_[{}, {})'.format(variable_name, bin_boundary[0], bin_boundary[1]))

        consolidated_bin_label = ' <OR> '.join(bin_labels)

        return consolidated_bin_label
