from gym.spaces.tuple_space import Tuple
import numpy as np
class Product(Tuple):
    @property
    def shape(self):
        #assume every action in space has a
        return (np.sum([c.shape[-1] for c in self.spaces]),)#(len(self.spaces), self.spaces[0].shape[-1])

    @property
    def shapes(self):
        # assume every action in space has a
        return [c.shape[-1] for c in self.spaces]

    @property
    def flat_dim(self):
        return np.sum([c.shape[-1] for c in self.spaces])

    def flatten(self, x):
        return np.concatenate([c.flatten(xi) for c, xi in zip(self.spaces, x)])

    def unflatten(self, x):
        dims = [c.shape[-1] for c in self.spaces]
        flat_xs = np.split(x, np.cumsum(dims)[:-1])
        return tuple(c.unflatten(xi) for c, xi in zip(self.spaces, flat_xs))
