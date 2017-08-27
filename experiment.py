import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_predict
from clustering import clustering
from clustering import eval_clustering_quality
from data_handler import DataHandler


class Experiment:
    
    def __init__(self, Metadata):
        self.file_path = Metadata.file_path
        try:
            self.data = pd.read_csv(self.file_path)
        except:
            raise ValueError('Failed to Read CSV as Pandas DataFrame')
        self.class_variable_name = Metadata.class_variable_name
        self.numerical_variable_names = Metadata.numerical_variable_names
        self.categorical_variable_names = Metadata.categorical_variable_names
        self.input_variable_names = self.numerical_variable_names + self.categorical_variable_names
        self.class_label = self.data[self.class_variable_name]
        self.number_of_class_type = len(np.unique(self.class_label))
        
        self.data_handler = DataHandler(self.data[self.input_variable_names], 
                           numerical_variable_names=self.numerical_variable_names,
                           categorical_variable_names=self.categorical_variable_names)
    
    def prepare_representations_of_data(self):
        self.dummy_coded_data = self.data_handler.dummy_coding_data()
        self.normalized_data = self.data_handler.normalize(self.dummy_coded_data)
        self.uniformly_binned_data = self.data_handler.uniformly_binning_data()
        self.semantically_binned_data = self.data_handler.semantically_binning_data()
    
    def eval_clustering_quality_of_each_representation(self, num_cluster=5,
                                                       clustering_method='kmeans'):
        print('>>>>> Clustering Quality >>>>>')
        label_raw = clustering(self.dummy_coded_data, num_cluster, clustering_method)
        label_normalized = clustering(self.normalized_data, num_cluster, clustering_method)
        label_uniform = clustering(self.uniformly_binned_data, num_cluster, clustering_method)
        label_semantic = clustering(self.semantically_binned_data, num_cluster, clustering_method)
        
        print('Raw: ', eval_clustering_quality(label_raw, self.class_label))
        print('Normalized: ', eval_clustering_quality(label_normalized, self.class_label))
        print('Uniformly Binned: ', eval_clustering_quality(label_uniform, self.class_label))
        print('Semantically Binned: ', eval_clustering_quality(label_semantic, self.class_label))
    
    def eval_classification_performance_of_each_representation(self):
        print('>>>>> Classification Performance >>>>>')
        
        clf = LogisticRegression()
        
        predicted = cross_val_predict(clf, self.dummy_coded_data, self.class_label, cv=10)
        print('Raw: ', accuracy_score(self.class_label, predicted))
        
        predicted = cross_val_predict(clf, self.normalized_data, self.class_label, cv=10)
        print('Normalized: ', accuracy_score(self.class_label, predicted))
        
        predicted = cross_val_predict(clf, self.uniformly_binned_data, self.class_label, cv=10)
        print('Uniformly Binned: ', accuracy_score(self.class_label, predicted))
        
        predicted = cross_val_predict(clf, self.semantically_binned_data, self.class_label, cv=10)
        print('Semantically Binned: ', accuracy_score(self.class_label, predicted))

    def perform_experiment(self):
        self.prepare_representations_of_data()
        self.eval_clustering_quality_of_each_representation()
        self.eval_classification_performance_of_each_representation()
        
    def print_number_of_variable_by_type(self):
        print('#Numerical Variables: {}'.format(len(numerical_variable_names)))
        print('#Categorical Variables: {}'.format(len(categorical_variable_names)))