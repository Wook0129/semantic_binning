import numpy as np
import pandas as pd


class DataHandler:
        
    def __init__(self, data, var_dict, normalize=False):
        
        self.categorical_vars = data[var_dict['categorical_vars']].astype(str)
        self.numerical_vars = data[var_dict['numerical_vars']].astype(np.float32)
        self.class_var = data[var_dict['class_var']]
        self.n_variables = self.categorical_vars.shape[1] + self.numerical_vars.shape[1]
        self.input_vars = var_dict['categorical_vars'] + var_dict['numerical_vars']
        
        if normalize:
            mean, std = self.numerical_vars.mean(), self.numerical_vars.std()
            self.numerical_vars = (self.numerical_vars - mean) / std
        
    def get_dummy_coded_data(self, init_discretize_method='equal_freq', 
                             n_init_bins=20, bins_by_variable=None):
        
        numerical_vars = self.numerical_vars.copy()
        categorical_vars = pd.get_dummies(self.categorical_vars.copy())
        
        if not bins_by_variable:
            if init_discretize_method == 'equal_width':
                for var in self.numerical_vars.columns:
                    numerical_vars[var] = pd.cut(numerical_vars[var], bins=n_init_bins)

            if init_discretize_method == 'equal_freq':
                for var in self.numerical_vars.columns:
                    numerical_vars[var] = pd.qcut(numerical_vars[var], q=n_init_bins)
        else:
            for var in bins_by_variable:
                
                is_numerical_var = 'split_point' in bins_by_variable[var]
                
                if is_numerical_var:
                    bins = bins_by_variable[var]['split_point']
                    numerical_vars[var] = pd.cut(numerical_vars[var], bins=bins)
                
                else:
                    for merged_bin in bins_by_variable[var]['merged_bins']:
                        cols = ['{}_{}'.format(var, x) for x in merged_bin.split(' <OR> ')]
                        if len(cols) >= 2:
                            categorical_vars[merged_bin] = categorical_vars[cols].sum(axis=1)
                            categorical_vars.drop(cols, axis=1, inplace=True)
                    
        numerical_vars = pd.get_dummies(numerical_vars)
    
        return pd.concat([categorical_vars, numerical_vars], axis=1)
