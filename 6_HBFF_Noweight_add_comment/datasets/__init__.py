from .market1501 import Market1501
from .dukemtmc import DukeMTMC
from .msmt17 import MSMT17
from .bases import ImageDataset, BaseDataset
from .sampler import RandomIdentitySampler

__all__ = ['init_dataset', 'ImageDataset', 'BaseDataset', 'RandomIdentitySampler', 'Market1501', 'DukeMTMC', 'MSMT17']

def init_dataset(name, **kwargs):
    """
    Dataset factory function
    """
    if name == 'market1501':
        return Market1501(**kwargs)
    elif name == 'dukemtmc':
        return DukeMTMC(**kwargs)
    elif name == 'msmt17':
        return MSMT17(**kwargs)
    else:
        raise KeyError(f"Unknown dataset: {name}")