import tensorflow as tf
import numpy as np
from research.models.model import Model
from abc import ABCMeta, abstractmethod
from research.id_estimate import twonn2

class EstimatorBase(metaclass=ABCMeta):
    def __init__(self, model_name: str, model: tf.keras.Model, img: tf.data.Dataset,
                 gpu: bool, verbose: bool = False):
        self.model_name = model_name
        self.model = model
        self.img = img
        self._selected_layer = []
        self.gpu = gpu
        self.verbose = verbose

    @abstractmethod
    def _preprocess(self):
        pass
    @abstractmethod
    def _postprocess(self):
        pass
    @abstractmethod
    def calculate(self):
        pass

    def get_output_by_featured_layers(self) -> list:

        featured_layers = self._get_featured_layers()
        layer_outputs=[]

        for layer in self.model.layers:
            if layer.name in featured_layers:
                self._selected_layer.append(layer.name)
                layer_outputs.append(layer.output)

        intermediate_model = tf.keras.Model(inputs=self.model.layers[0].input, outputs=layer_outputs)
        outputs = intermediate_model.predict(self.img)
        return outputs if isinstance(outputs, list) else [outputs]

    def _get_featured_layers(self):
        featured_layers_dict = dict({ # exclude input
            Model.VGG16:['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool'],
            Model.VGG19:['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool'],
            Model.ResNet50:['conv2_bloc3_out', 'conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out'],
            Model.InceptionV3:[],
        })
        featured_layers = featured_layers_dict[Model.from_name(self.model_name)]
        featured_layers.append(self.model.layers[-1].name)
        return featured_layers

class Twonn2Estimator(EstimatorBase):
    def __init__(self, model_name, model, img, gpu, verbose):
        super().__init__(model_name, model, img, gpu, verbose)

    def _preprocess(self, outputs: list):

        return outputs
    def _postprocess(self, outputs: list):
        return dict(zip(self._selected_layer, outputs))

    def calculate(self, outputs: list, decompose_estimate: int):
        outputs = self._preprocess(outputs)
        res = []
        for output in outputs:
            val = twonn2.estimate_id(output, gpu=self.gpu,
                                     decompose_estimate=decompose_estimate,
                                     verbose=self.verbose)
            res.append(val)
            print(val)
        res = self._postprocess(res)
        return res

    def inference_and_calculate(self, dataset: tf.data.Dataset):
        featured_layers = self._get_featured_layers()
        res = []
        selected_layers = []

        for layer in self.model.layers:
            if layer.name in featured_layers:
                selected_layers.append(layer.name)

                intermediate_model = tf.keras.Model(inputs=self.model.layers[0].input, outputs=layer.output)
                val = twonn2.estimate_id_from_dataset(dataset, intermediate_model,
                                                      gpu = self.gpu, verbose=self.verbose)
                res.append(val)
                print(val)

        res = dict(zip(selected_layers, res))
        return res


