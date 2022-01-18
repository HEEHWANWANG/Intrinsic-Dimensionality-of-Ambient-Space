import tensorflow as tf
import tensorflow_datasets as tfds
from research.models.dataset import Dataset
from research.models.model import Model

class DatasetLoader(object):
    def __init__(self, dataset_name:str,
                       dataset_dir:str,
                       batch_size:int,
                       model_name:str,
                       sample_ratio:int,
                       target_size:int,
                       train_valid_split: tuple = (80,20),
                       verbose: bool = False,
                ):
        if self._is_valid_ratio is False:
            raise ValueError(f'sum of ratio should be 100, but sum of input is {sum(train_valid_split)}')

        self.dataset = Dataset.from_name(dataset_name).value
        self.model_name = Model.from_name(model_name).value

        self.train_ratio, self.valid_ratio = train_valid_split
        self.sample_ratio = int(sample_ratio)
        self.target_size = target_size
        self.batch_size = batch_size
        self.data_dir = dataset_dir
        self.verbose = verbose

        self.dataset_info = dict()
        if self.verbose:
            print("DatasetLoader.__init__:", locals())
            print("DatasetLoader.dataset_info:", self.dataset_info)

    def _is_valid_ratio(self) -> bool:
        return sum(self.train_valid_split)==100

    def load_dataset(self) -> tuple:
        _ratio = int(self.train_ratio*self.sample_ratio/100)
        (train_ds, valid_ds, test_ds), ds_info = tfds.load(name=self.dataset,
                                                           split=[f'train[:{_ratio}%]',f'train[{_ratio}%:{self.sample_ratio}%]', f'test[:{self.sample_ratio}%]'],
                                                           data_dir = self.data_dir, with_info=True, shuffle_files = True, as_supervised=True)

        self._parse_dataset_info(ds_info)
        train_ds = train_ds.shuffle(10000).map(self._preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        valid_ds = valid_ds.shuffle(10000).map(self._preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        test_ds = test_ds.map(self._preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(self.batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        return (train_ds, valid_ds, test_ds)

    def _parse_dataset_info(self, ds_info):
        self.dataset_info["image_shape"] = ds_info.features["image"].shape
        self.dataset_info["num_classes"] = ds_info.features["label"].num_classes
        self.dataset_info["size"] = dict()
        self.dataset_info["size"]["train"] = int(ds_info.splits["train"].num_examples*(self.train_ratio/100))
        self.dataset_info["size"]["valid"] = int(ds_info.splits["train"].num_examples*(self.valid_ratio/100))
        self.dataset_info["size"]["test"] = ds_info.splits["test"].num_examples
        self.dataset_info["raw"] = ds_info

        if self.target_size == None:
            original_size_for_dataset = {
                "cifar10": 32,
                "caltech_birds2011": 150,
            }
            assert self.dataset in original_size_for_dataset.keys()
            self.target_size = original_size_for_dataset[self.dataset]


    def get_dataset_info(self):
        return self.dataset_info

    def _preprocess(self, image, label):
        image = tf.image.resize(image, (self.target_size, self.target_size), method=tf.image.ResizeMethod.BICUBIC)
        module_path = "tensorflow.keras.applications."+self.model_name
        image = getattr(__import__(module_path, fromlist=[self.model_name]), 'preprocess_input')(image)

        label = tf.one_hot(label, self.get_dataset_info()["num_classes"])

        return image, label




