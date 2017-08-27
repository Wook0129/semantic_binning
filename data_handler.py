import numpy as np
import pandas as pd
from uniform_binning import UniformBinning
from semantic_binning import SemanticBinning


class DataHandler:
        
    def __init__(self, data, *, categorical_variable_names, numerical_variable_names):
        
        def set_type_of_variables(data, categorical_variable_names, numerical_variable_names):
            categorical_data = data[categorical_variable_names].astype(str)
            data[categorical_variable_names] = categorical_data
            numerical_data = data[numerical_variable_names].astype(np.float32)
            data[numerical_variable_names] = numerical_data
        
        if type(data) == pd.DataFrame:
            self.data = data
        else:
            raise ValueError('Type of data should be Pandas DataFrame')
            
        if (type(categorical_variable_names) == list) and (type(numerical_variable_names) == list):
            self.categorical_variable_names = categorical_variable_names
            self.numerical_variable_names = numerical_variable_names
            self.input_variable_names = categorical_variable_names + numerical_variable_names
        else:
            raise ValueError('Type of categorical/numerical_variable_names should be list')

        set_type_of_variables(self.data, self.categorical_variable_names,
                                         self.numerical_variable_names)

    def dummy_coding_data(self):
        numerical_data = self.data[self.numerical_variable_names]
        categorical_data = pd.get_dummies(self.data[self.categorical_variable_names])
        dummy_coded_data = pd.concat([numerical_data, categorical_data], axis=1)
        return dummy_coded_data

    def uniformly_binning_data(self, bin_numbers=5):
        if (type(bin_numbers) != int) or bin_numbers < 2:
            raise ValueError('bin_numbers should be at least 2')

        uniform_binner = UniformBinning(bin_numbers=bin_numbers)
        uniformly_binned_data = uniform_binner.binning(self.data, self.numerical_variable_names)
        
        return uniformly_binned_data
    
    def semantically_binning_data(self, uniform_bin_numbers=20):
        
        semantic_binner = SemanticBinning(uniform_bin_numbers=uniform_bin_numbers)
        semantically_binned_data = semantic_binner.binning(self.data, self.numerical_variable_names)
        
        return semantically_binned_data
    
    def normalize(self, data, method='mean_std'):
        if type(data) != pd.DataFrame:
            raise ValueError('Type of data should be Pandas DataFrame')
        if method == 'mean_std':
            return (data-data.mean())/data.std()
        elif method == 'min_max':
            return (data-data.min())/(data.max()-data.min())
        else:
            raise NotImplementedError
