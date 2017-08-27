from collections import namedtuple


Metadata = namedtuple('Metadata',
                     'file_path class_variable_name numerical_variable_names '+
                     'categorical_variable_names')