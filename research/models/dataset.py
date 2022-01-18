from enum import Enum

class Dataset(Enum):
    '''tfds dataset name'''
    CIFAR10 = 'cifar10'
    MNIST = 'mnist'
    CUB = 'caltech_birds2011'

    @classmethod
    def from_name(cls, name):
        for ds, ds_name in cls.get_dict().items():
            if ds_name == name:
                return ds
        raise ValueError('{} is not a valid dataset name'.format(name))
    
    @classmethod
    def get_dict(cls): #alias
        return {Dataset.CIFAR10: 'cifar10',
                Dataset.MNIST: 'mnist',
                Dataset.CUB: 'cub200',
                }