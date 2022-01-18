import os, sys
sys.path.append(os.getcwd())

import datetime
import tensorflow as tf

from research.config import ArgParser
from research.data_loader import DatasetLoader
from research.model_builder import ModelBuilder
from research.id_estimator import Twonn2Estimator
from research.utils import save_result

def main():
    """parse arguments (from command line)"""
    args = ArgParser().parse_args()

    # use_loaded_model=False
    use_loaded_model=True
    ################################
    """modify this for training"""
    ################################
    target_size = 150
    learning_rate = 1e-3
    epochs = 10
    batch_size = 64

    ################################
    #modify this for saving results#
    ################################
    checkpoint_saved_name = f'{args.dataset}_{args.model}_uf_{len(args.layers)}_size_{target_size}'
    # model_saved_name = os.path.join(args.base_dir, 'results', checkpoint_saved_name)
    model_saved_name = os.path.join(args.base_dir, 'ckpt', checkpoint_saved_name)
    ################################


    """load dataset"""
    dataset_loader = DatasetLoader(
        dataset_name=args.dataset, dataset_dir = os.path.join(args.base_dir, 'datasets'),
        batch_size = batch_size, target_size = target_size,
        model_name = args.model, sample_ratio=args.sample_ratio,
        verbose = args.verbose)
    train_ds, valid_ds, test_ds = dataset_loader.load_dataset()

    dataset_num_classes = dataset_loader.get_dataset_info()["num_classes"]
    input_shape = dataset_loader.get_dataset_info()["image_shape"]

    """get available gpu (if not available, set cpu)"""
    device_list = []
    if args.gpu:
        device_list = [f'/gpu:{i}' for i in range(len(tf.config.experimental.list_physical_devices("GPU")))]
    print(device_list)
    distribution = tf.distribute.MirroredStrategy(device_list)

    with distribution.scope():

        if not use_loaded_model:
            """build base model & add last layer"""
            model_builder = ModelBuilder(model_name=args.model, input_shape = (target_size, target_size,3))
            model_builder.load_pretrained_model()
            model_builder.freeze_all_layer(_compile=False)
            model_builder.add_classifier(num_classes = dataset_num_classes)
            # model_builder.print_layer_names()
            model = model_builder.unfreeze_by_layer_name(args.layers, _compile=False)
            print(model.summary())

            """train model"""
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, decay=0.01, nesterov=True)
            loss = tf.keras.losses.CategoricalCrossentropy()
            model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
            # checkpoint_filepath = f'./checkpoint/{epoch:02d}-{val_loss:.5f}.h5'
            checkpoint_filepath = f'./results/checkpoint/ckp_{checkpoint_saved_name}.h5'
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,
                save_weights_only=True,
                monitor='val_accuracy',
                mode='auto',
                save_best_only=True,
                verbose=1)
            history=model.fit(train_ds, epochs=epochs, validation_data=valid_ds, callbacks=[early_stopping, model_checkpoint])
            # print("history dict: ", history.history)

            loss, accuracy = model.evaluate(test_ds)
            print("loss: ", loss, "accuracy: ", accuracy)
            model.save(model_saved_name)


        else:
            loaded_model = tf.keras.models.load_model(model_saved_name)
            print(loaded_model.summary())
            model = loaded_model
            # model.load_weights(checkpoint_path)


        """calculate intrinsic dimension from each layers(called as featured layers)"""
        img = test_ds # or train?
        if args.method=='twonn2':
            estimator = Twonn2Estimator(model_name=args.model, model=model, img=img,
                                        gpu=args.gpu, verbose=args.verbose)
        # elif args.method=='mle'
        # elif args.method=='geomle'
        if args.estimate_batch:
            dataset_loader = DatasetLoader(
                dataset_name=args.dataset, dataset_dir = os.path.join(args.base_dir, 'datasets'),
                batch_size = args.estimate_batch, target_size = target_size,
                model_name = args.model, sample_ratio=args.sample_ratio,
                verbose = args.verbose)
            train_ds, valid_ds, test_ds = dataset_loader.load_dataset()
            results = estimator.inference_and_calculate(test_ds)
        else:
            outputs_by_layers = estimator.get_output_by_featured_layers()
            results = estimator.calculate(outputs_by_layers, args.decompose_estimate)
        print(results)

        save_args = ['model', 'dataset', 'sample_ratio', 'method', 'layers']
        result_dict = {k: args.__dict__[k] for k in save_args}
        result_dict.update(
            {'tag': args.tag,
             'datetime': datetime.datetime.now().strftime('%x %X')})
        result_dict.update(results)

        save_result(result_dict, verbose=args.verbose)

    return
if __name__== '__main__':
    main()
