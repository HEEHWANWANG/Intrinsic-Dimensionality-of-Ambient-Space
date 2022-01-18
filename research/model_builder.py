from research.models.model import Model
import tensorflow as tf
from tensorflow import keras
# import keras
# from keras.layers import Dense

class ModelBuilder(object):
    def __init__(self, model_name:str, input_shape):
        self.model = None
        self.model_name = model_name
        self.input_shape = input_shape
        self._sequential = self._check_sequential()

    def load_pretrained_model(self):
        module_path = "tensorflow.keras.applications."+self.model_name
        base_model = getattr(__import__(module_path, fromlist=[self.model_name]), Model.from_name(self.model_name).name)
        base_model = base_model(include_top=False, weights='imagenet', input_shape=self.input_shape)
#        self.model = base_model(include_top=False, weights='imagenet', input_shape=self.input_shape)

#        self.model = keras.Sequential(layers = self.model.layers, name='base_model') if self._sequential else base_model
        self.model = keras.Sequential(layers = base_model.layers, name='base_model') if self._sequential else base_model

        return self.model

    def add_classifier(self, num_classes:int):

        if self._sequential:
            self.model.add(keras.layers.GlobalAveragePooling2D())
            self.model.add(keras.layers.Dense(num_classes, activation='softmax'))
        else:
            output = keras.layers.GlobalAveragePooling2D()(self.model.output)
            output = keras.layers.Flatten(name="flatten")(output)
            output = keras.layers.Dense(256, activation="relu")(output)
            output = keras.layers.Dropout(0.5)(output)
            output = keras.layers.Dense(num_classes, activation="softmax")(output)
            self.model = keras.Model(inputs=self.model.input, outputs=output)

        return self.model

    def freeze_all_layer(self, _compile=True):
        for layer in self.model.layers:
            layer.trainable = False

        if _compile:
            self._compile()
        return self.model

    def freeze_until_layer(self, freeze_layer: str, _compile=True):
        if not freeze_layer:
            return self.model
        else:
            assert freeze_layer in self.model.layer

        for layer in self.model.layers:
            if layer.name == layer: break
            layer.trainable = False

        if _compile:
            self._compile()
        return self.model

    def unfreeze_by_layer_name(self, layer_names: list, _compile=True):
        # unfreeze
        for layer in self.model.layers:
            for layer_name in layer_names:
                if layer_name==layer.name: layer.trainable=True
        # compile
        if _compile:
            self._compile()

        return self.model

    def _compile(self):
        self.model.compile()

    def _check_sequential(self,):
        return False if self.model_name.startswith("res") else True

    def print_layer_names(self):
        print(list(map(lambda layer: layer.name, self.model.layers)))
