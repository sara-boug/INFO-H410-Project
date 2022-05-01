import argparse

import source.config as config
from source.generate_data import TrainingDataGenerator
from source.trainer import SegNetTrainer
from source.segnet_model import SegNet

import tensorflow as tf


class ModeTrainer:

    @staticmethod
    def generate_data():
        data_generator = TrainingDataGenerator(images_path=config.images_dataset_path,
                                               masks_path=config.masks_dataset_path)
        data_generator.execute()

    @staticmethod
    def train_segnet():
        segnet_trainer = SegNetTrainer(train_data_path=config.training_set_path,
                                       val_data_path=config.validation_set_path)
        model = SegNet(config.num_classes, config.network_input_shape).get_model()
        loss_func = tf.keras.losses.CategoricalCrossentropy()
        accuracy_func = tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy", dtype=None)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        segnet_trainer.train(model=model, loss_func=loss_func, accuracy_func=accuracy_func,optimizer=optimizer)


if __name__ == "__main__":
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true', required=False)
    parser.add_argument('--train', action='store_true', required=False)

    arguments = parser.parse_args()
    model_trainer = ModeTrainer()
    cuda_num = 0

    if arguments.preprocess:
        model_trainer.generate_data()
    if arguments.train:
        model_trainer.train_segnet()
