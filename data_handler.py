import numpy as np
import pandas as pd


class DataHandler:
        
    def __init__(self, data, var_dict):
        
        self.var_dict = var_dict
        
        if 'categorical_vars' in var_dict:
            self.categorical_vars = data[var_dict['categorical_vars']].astype(str)
            self.n_variables = len(var_dict['numerical_vars'] + var_dict['categorical_vars'])      
        else:
            self.categorical_vars = None
            self.n_variables = len(var_dict['numerical_vars'])
            
        self.numerical_vars = data[var_dict['numerical_vars']].astype(np.float32)
        self.class_var = data[var_dict['class_var']]
        
    def get_dummy_coded_data(self, init_discretize_method='equal_freq', 
                             n_init_bins=20, bins_by_variable=None):
        
        numerical_vars = self.numerical_vars.copy()
        
        if 'categorical_vars' in self.var_dict:
            categorical_vars = pd.get_dummies(self.categorical_vars.copy())
        
        if not bins_by_variable:
            
            if init_discretize_method == 'equal_width':
                for var in self.numerical_vars.columns:
                    numerical_vars[var] = pd.cut(numerical_vars[var], bins=n_init_bins)
                    
            elif init_discretize_method == 'equal_freq':
                
                for var in self.numerical_vars.columns:
                    
                    quantized = pd.qcut(numerical_vars[var], q=n_init_bins, duplicates='drop')
                    
                    if len(quantized.unique()) != n_init_bins:
                        jitter = np.random.normal(loc=0, scale=1e-10, 
                                                  size=len(numerical_vars[var]))
                        quantized = pd.qcut(numerical_vars[var]+jitter, q=n_init_bins, duplicates='drop')
                        
                    numerical_vars[var] = quantized
                    
            elif init_discretize_method == 'scale_numeric':
                mean, std = numerical_vars.mean(), numerical_vars.std()
                numerical_vars = (numerical_vars - mean) / std
                
            elif init_discretize_method == 'dummy_only':
                pass
            
            else:
                raise NotImplementedError
            
        else:
            for var in bins_by_variable:
                bins = bins_by_variable[var]['split_point']
                numerical_vars[var] = pd.cut(numerical_vars[var], bins=bins)
                    
        numerical_vars = pd.get_dummies(numerical_vars)
        
        if 'categorical_vars' in self.var_dict:
            return pd.concat([categorical_vars, numerical_vars], axis=1)
        else:
            return numerical_vars
    
    def get_bins_by_variable_from_data(self, dummy_coded_data):

        def get_variable_name(dummy_variable_name):
            return '_'.join(dummy_variable_name.split('_')[:-1])

        def get_interval_and_split_points(dummy_variable_name):
            interval = dummy_variable_name.split('_')[-1]
            begin = float(interval.split(', ')[0].replace('(',''))
            end = float(interval.split(', ')[1].replace(']',''))
            return interval, begin, end
    
        bins_by_variable = dict()

        for var in self.numerical_vars.columns:

            bins = []
            split_points = set()

            dummy_vars = [x for x in dummy_coded_data.columns 
                          if var == get_variable_name(x)]

            for dummy_var in dummy_vars:
                interval, begin, end = get_interval_and_split_points(dummy_var)
                bins.append(interval)
                split_points.update([begin, end])

            split_points = sorted(split_points)

            bins_by_variable[var] = dict(bins=bins, split_point=split_points)

        return bins_by_variable
