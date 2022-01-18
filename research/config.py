
import os, argparse
from tensorflow.test import is_gpu_available

from research.models.model import Model
from research.models.dataset import Dataset

class ArgParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(prog="id estimation", description="Argument parser")

        self.dataset_available = Dataset.get_dict().values()
        self.model_available = Model.get_dict().values()
        self.estimation_available = ['twonn2', 'twonn', 'mle', 'geomle']
        self.add_argument()

    def add_argument(self):
        # Model, dataset, id estimation tool
        self.parser.add_argument('-m', '--model', type=str, default='vgg16', choices=self.model_available,
                            help='Model')
        self.parser.add_argument('-d', '--dataset', type=str, default='cifar10', choices=self.dataset_available,
                            help='Dataset to feed')
        self.parser.add_argument('-sr', '--sample_ratio', type=str, default=100,
                            help='Sample size of dataset')
        self.parser.add_argument('-em', '--method', type=str, default='twonn2', choices=self.estimation_available,
                            help='ID Estimation Method')
        self.parser.add_argument('-de', '--decompose_estimate', type=int, nargs='?',
                            help='# of decomposed steps. Only provided for twonn2')
        self.parser.add_argument('-eb', '--estimate_batch', type=int, nargs='?',
                            help='batch-level estimation using tf.data.Dataset')
        self.parser.add_argument('-l', '--layers', type=str, nargs='+', default=[],
                            help='Layers to unfreeze. Only available with gpu')
        # other
        self.parser.add_argument('-bd', '--base_dir', type=str, default=os.getcwd(),
                            help='Where to store dataset, model and results')
        self.parser.add_argument('-g','--gpu', type=bool, default = False)
        self.parser.add_argument('-t','--tag', type=str, default = None)
        self.parser.add_argument('-v','--verbose', type=bool, default = False)

    def parse_args(self):
        """"validation"""
        rargs = self.parser.parse_args()
        if rargs.gpu and not self._is_gpu_available():
            print("GPU is not available in your environment. Option will be automatically turned off.")
            rargs.gpu = False
        # if len(rargs.layers)>0 and not self._is_gpu_available():
        #     print("GPU is not available in your environment. Option will be automatically turned off.")
        #     rargs.gpu = False

        return rargs

    def _is_gpu_available(self):
        return is_gpu_available()

