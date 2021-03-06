import argparse
import os

import source.config as config
from source.generate_data import TrainingDataGenerator
from source.trainer import SegNetTrainer
from source.segnet_model import SegNet
from source.test_model import TestModel
from source.metrics import multi_class_dice_score
import tensorflow as tf


class ModeTrainer:

    @staticmethod
    def generate_data():
        data_generator = TrainingDataGenerator(images_path=config.images_dataset_path,
                                               masks_path=config.masks_dataset_path)

        # Generate all the data /all
        data_generator.execute()
        # Generate the data per set
        data_generator.generate_files_per_set()

    @staticmethod
    def get_model_performance():
        model_tester = TestModel(config.model_path)
        model_tester.compute_model_performance_perclass()

    @staticmethod
    def test_segnet(image):
        model_tester = TestModel(config.model_path)
        image_name = image + ".nii"
        mask_name = image + "_segmentation.nii"

        image_path = os.path.join(config.test_set_path, image_name)
        mask_path = os.path.join(config.test_set_path, mask_name)

        model_tester.predict(image_path, mask_path)

    @staticmethod
    def show_evolution(path_to_metrics):
        TestModel.display_training_evolution(path_to_metrics)

    def train_segnet(self):
        segnet_trainer = SegNetTrainer(train_data_path=config.training_set_path,
                                       val_data_path=config.validation_set_path)
        model = SegNet(config.num_classes, config.network_input_shape).get_model()
        loss_func = multi_class_dice_score
        accuracy_func = tf.keras.metrics.MeanIoU(num_classes=5)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        segnet_trainer.train(model=model,
                             loss_func=loss_func, accuracy_func=accuracy_func, optimizer=optimizer)


if __name__ == "__main__":
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true', required=False)
    parser.add_argument('--train', action='store_true', required=False)
    parser.add_argument('--performance', action='store_true', required=False)
    parser.add_argument('--show_evol', action='store_true', required=False)
    parser.add_argument('--predict', type=str, required=False)

    arguments = parser.parse_args()
    model_trainer = ModeTrainer()
    cuda_num = 0

    if arguments.preprocess:
        model_trainer.generate_data()
    if arguments.train:
        model_trainer.train_segnet()
    if arguments.predict:
        image_name = arguments.predict
        model_trainer.test_segnet(image_name)
    if arguments.show_evol:
        model_trainer.show_evolution(config.metrics_path)
    if arguments.performance:
        model_trainer.get_model_performance()
