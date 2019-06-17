import numpy as np

def to_one_hot(data, cls=None):
    if cls is None:
        cls = len(np.unique(data))
    sh = data.shape
    hot = np.zeros((np.prod(sh), cls), dtype=data.dtype)
    hot[np.arange(np.prod(sh)), data.flatten()] = 1
    return np.reshape(hot, sh + (cls, ))
