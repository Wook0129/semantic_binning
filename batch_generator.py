import numpy as np


class BatchGenerator:
    
    def __init__(self, inputs, targets, batch_size):
        self.inputs = np.array(inputs)
        self.targets = np.array(targets)
        self.batch_size = batch_size
        self.iter = self.make_random_iter()

    def make_random_iter(self):
        splits = np.arange(self.batch_size, len(self.inputs), self.batch_size)
        np.random.seed(42)
        it = np.split(np.random.permutation(range(len(self.inputs))), splits)[:-1]
        return iter(it)
        
    def next_batch(self):
        try:
            idxs = next(self.iter)
        except StopIteration:
            self.iter = self.make_random_iter()
            idxs = next(self.iter)
        
        return self.inputs[idxs], self.targets[idxs]
