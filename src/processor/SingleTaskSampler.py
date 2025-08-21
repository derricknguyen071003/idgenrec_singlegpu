from torch.utils.data import Sampler
import random
import math

class SimpleSampler(Sampler):
    def __init__(self, data_source, batch_size, seed=42, shuffle=True):
        self.data_source = data_source
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self):
        indices = list(range(len(self.data_source)))
        if self.shuffle:
            random.seed(self.seed + self.epoch)
            random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch):
        self.epoch = epoch
