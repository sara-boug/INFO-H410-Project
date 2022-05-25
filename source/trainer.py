import math
import os
import logging

from source.config import epochs, batch_size, num_classes, model_path, metrics_path

import tensorflow as tf
import numpy as np
import SimpleITK as sitk
import pandas as pd


class SegNetTrainer:
    def __init__(self, train_data_path, val_data_path):
        """

        :param train_data_path: path to the folder containing the training set
        :param val_data_path: path to th folder containing the validation set
        """
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path

    def train(self, model, loss_func, accuracy_func, optimizer):
        """
         Trains the model

        :param model: Segnet model
        :param loss_func: The loss function for the training (used a criterion for the back propagation)
        :param accuracy_func: The accuracy metric for the validation
        :param optimizer: The model optizer
        :return:
        """
        # 1 : Extract the training set and validation file names
        # both images and masks have (batch_size, all_batches) shape
        train_images, train_masks = self.get_files(path=self.train_data_path)

        val_images, val_masks = self.get_files(path=self.val_data_path)
        # 2: Define the dataframe  that will contain the model metrics
        metrics_df = pd.DataFrame([], columns=['epoch', 'train_loss', 'val_loss', 'accuracy'])
        metrics_df.to_csv(metrics_path)

        # metrics
        training_losses = []
        validation_losses = []
        accuracies = []

        for epoch in range(epochs):
            # Per epoch the metrics arrays should be cleared
            training_losses.clear()
            validation_losses.clear()
            accuracies.clear()

            # training
            for f_image, f_mask in zip(train_images, train_masks):
                # mini batch is used for the training
                image_batch = self.__extract_batch_data_array(f_image, True)
                mask_batch = self.__extract_batch_data_array(f_mask, True)
                # Since this is multiclass classification
                mask_batch = tf.keras.utils.to_categorical(mask_batch, num_classes=num_classes, dtype='float32')

                with tf.GradientTape() as tape:
                    sfmx_logits = model(image_batch, training=True)  # Softmax output probabilities
                    loss = loss_func(mask_batch, sfmx_logits)

                    training_losses.append(loss)

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # validation
            for f_image, f_mask in zip(val_images, val_masks):
                image_batch = self.__extract_batch_data_array(f_image, False)

                mask_batch = self.__extract_batch_data_array(f_mask, False)
                mask_batch = tf.keras.utils.to_categorical(mask_batch, num_classes=num_classes, dtype='float32')
                sfmx_logits = model(image_batch, training=False)  # Softmax output probabilities
                loss = loss_func(mask_batch, sfmx_logits)  # compute the loss
                validation_losses.append(loss)
                accuracy = accuracy_func(mask_batch, sfmx_logits)  # compute the accuracy
                accuracies.append(accuracy)

            self.__write_to_csv(training_losses, validation_losses, accuracies, metrics_df, epoch)
            model.save(model_path)

    @staticmethod
    def __write_to_csv(training_losses, validation_losses, accuracies, dataframe, epoch):
        tmean = np.mean(training_losses)  # training loss mean
        vmean = np.mean(validation_losses)  # validation loss mean
        amean = np.mean(accuracies)  # accuracies mean

        losses_df = dataframe.append({'epoch': epoch,
                                      'train_loss': tmean,
                                      'val_loss': vmean,
                                      'accuracy': amean,
                                      }, ignore_index=True)
        losses_df.to_csv(metrics_path, header=False, mode='a')
        logging.info(f"train loss : {tmean}")
        logging.info(f"validation loss : {vmean}")
        logging.info(f"accuracy : {amean}")

    @staticmethod
    def get_files(path, with_batch=False) -> np.array:
        """
         Extracts the files  from the given path

        A tuple is returned such as [[image1,mask1],[images2, mask2]....]
        :param with_batch:
        :param path: path to the folder
        :return:
        """
        files = os.listdir(path)
        files.sort()  # order is important !
        files = np.reshape(files, (len(files) // 2, -1))
        images = files[:, 0]
        masks = files[:, 1]
        assert images.shape[0] == masks.shape[0]

        if with_batch:
            intended_size = math.floor(images.shape[0] / batch_size + 1)

            # Now the data will be partitioned according to the batch size
            images = np.array_split(images, intended_size, )

            masks = np.array_split(masks, intended_size)
            return images, masks
        else :
            return images, masks

    def __extract_batch_data_array(self, file_names, is_training: bool):
        """
        Reads the files and returns an array containing the image batch

        :param file_names:
        :param is_training: checks whether the model is training
               if it is the case, the reading folder the training set
        :return:
        """
        all_data = None

        root_folder = self.train_data_path
        if not is_training:
            root_folder = self.val_data_path
        for file_name in file_names:
            path = os.path.join(root_folder, file_name)
            image = sitk.GetArrayFromImage(sitk.ReadImage(path))  # image size = [h,w]

            image = np.expand_dims(image, axis=-1)  # for the channel [h,w,1]
            image = np.expand_dims(image, axis=0)  # for the batch [1,h,w,1]

            if all_data is None:
                all_data = image
            else:
                all_data = np.concatenate([all_data, image], axis=0)

        return all_data
