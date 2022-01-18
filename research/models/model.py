from enum import Enum

class Model(Enum):
    '''package name under tf.keras.applications'''
    VGG16 = "vgg16"
    VGG19 = "vgg19"
    ResNet50 = "resnet50"
    InceptionV3 = "inception_v3"

    @classmethod
    def from_name(cls, name):
        for model, model_name in cls.get_dict().items():
            if model_name == name:
                return model
        raise ValueError('{} is not a valid model name'.format(name))
    
    @classmethod
    def get_dict(cls): #alias
        return {Model.VGG16: 'vgg16',
                Model.VGG19: 'vgg19',
                Model.ResNet50: 'resnet50',
                Model.InceptionV3: 'inception',
                }

