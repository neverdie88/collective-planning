from gym.spaces import box
import numpy as np

class Box(box.Box):
    def flatten(self, x):
        return np.asarray(x).flatten()

    def unflatten(self, x):
        return np.asarray(x).reshape(self.shape)