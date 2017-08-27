import numpy as np
import pandas as pd


class UniformBinning:
    
    def __init__(self, *, bin_numbers=5):
        
        if (type(bin_numbers) != int) or bin_numbers < 2:
            raise ValueError('bin_numbers should be at least 2')
            
        self.bin_numbers = bin_numbers
        
    def binning(self, data, numerical_variable_names):
        
        uniformly_binned_data = data.copy()
        
        for variable_name in numerical_variable_names:
            bin_boundaries = self._uniformly_discretize_numerical_variable(uniformly_binned_data[variable_name], self.bin_numbers)
            bin_labels = self._make_bin_labels(bin_boundaries)
            uniformly_binned_data[variable_name] = uniformly_binned_data[variable_name].apply(lambda x: self._put_into_bin(x, bin_boundaries, bin_labels))

        uniformly_binned_data = pd.get_dummies(uniformly_binned_data)
        
        return uniformly_binned_data
    
    def _uniformly_discretize_numerical_variable(self, variable_values, bin_numbers):

        if not isinstance(variable_values, pd.Series):
            raise ValueError('Expect Pandas Series for variable_values at _uniformly_discretize_numerical_variable()')

        min_value = variable_values.min()
        max_value = variable_values.max()
        bin_size = (max_value - min_value) / bin_numbers

        bin_boundaries = [min_value] + [min_value + i*bin_size for i in range(1,bin_numbers)] + [max_value]

        return bin_boundaries

    def _make_bin_labels(self, bin_boundaries):

        bin_labels = ['{:.3g})'.format(bin_boundaries[1])]

        for index in range(1, len(bin_boundaries) - 2):
            bin_label = '[{:.3g}, {:.3g})'.format(bin_boundaries[index], bin_boundaries[index+1])
            bin_labels.append(bin_label)

        bin_labels.append('[{:.3g}'.format(bin_boundaries[-2]))

        return bin_labels

    def _put_into_bin(self, x, bin_boundaries, bin_labels):            

        if x < bin_boundaries[1]:
            return bin_labels[0]
        elif x >= bin_boundaries[-2]:
            return bin_labels[-1]
        else:
            for index in range(1, len(bin_boundaries) - 2):
                if bin_boundaries[index] <= x < bin_boundaries[index+1]:
                    bin_for_x = bin_labels[index]
                    return bin_for_x
