import argparse

import source.config as config
from source.generate_data import TrainingDataGenerator
from source.trainer import SegNetTrainer
from source.segnet_model import SegNet
from source.test_model import  TestModel

import tensorflow as tf
import keras.backend as K


class ModeTrainer:

    @staticmethod
    def generate_data():
        data_generator = TrainingDataGenerator(images_path=config.images_dataset_path,
                                               masks_path=config.masks_dataset_path)
        #data_generator.execute()
        data_generator.generate_files_per_set()

    def test_segnet(self):
        model_tester = TestModel(config.model_path)
        image_path = "/home/sara/Documents/techniques_of_artificial_intelligence/INFO-H410-Project/data" \
                     "/preprocessed_data/training_set/image125.nii "
        predict = model_tester.predict(image_path)

    @staticmethod
    def multi_class_dice_score(ground_truth, prediction):
        def _dice_index(ground_truth, prediction, smooth=1e-6):
            # flatten label and prediction tensors
            targets = K.flatten(ground_truth)
            inputs = K.flatten(prediction)
            intersection = K.sum(targets * inputs)
            dice = (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
            return 1 - dice

        index = 0
        for i in range(0, ground_truth.shape[-1]):
            dice = _dice_index(ground_truth[:, :, :, i], prediction[:, :, :, i])
            index += dice

        return index / 5

    def train_segnet(self):
        segnet_trainer = SegNetTrainer(train_data_path=config.training_set_path,
                                       val_data_path=config.validation_set_path)
        model = SegNet(config.num_classes, config.network_input_shape).get_model()
        loss_func = self.multi_class_dice_score
        accuracy_func = tf.keras.metrics.MeanIoU(num_classes=5)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

        segnet_trainer.train(model=model,
                  loss_func=loss_func, accuracy_func=accuracy_func, optimizer=optimizer)


if __name__ == "__main__":
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true', required=False)
    parser.add_argument('--train', action='store_true', required=False)
    parser.add_argument('--test', action='store_true', required=False)

    arguments = parser.parse_args()
    model_trainer = ModeTrainer()
    cuda_num = 0

    if arguments.preprocess:
        model_trainer.generate_data()
    if arguments.train:
        model_trainer.train_segnet()
    if arguments.test:
        model_trainer.test_segnet()
