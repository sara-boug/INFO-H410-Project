import math
import os

from source.config import epochs, batch_size, image_size, num_classes

import tensorflow as tf
import numpy as np
import SimpleITK as sitk


class SegNetTrainer:
    def __init__(self, train_data_path, val_data_path):
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path

    def train(self, model, loss_func, optimizer):
        # both images and masks have (batch_size, all_batches) shape
        train_images, train_masks = self.__get_files(path=self.train_data_path)
        # val_ds = self.__get_files(path=self.val_data_path)
        for epoch in range(epochs):
            for f_image, f_mask in zip(train_images, train_masks):
                image_batch = self.__extract_batch_data_array(f_image, True)

                mask_batch = self.__extract_batch_data_array(f_mask, True)
                mask_batch = tf.keras.utils.to_categorical(mask_batch, num_classes=num_classes,dtype='float32')

                with tf.GradientTape() as tape:
                    sfmx_logits = model(image_batch, training=True)  # Softmax output probabilities
                    loss = loss_func(mask_batch, sfmx_logits)
                    print(loss)

                grads = tape.gradient(loss, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

    @staticmethod
    def __get_files(path) -> np.array:
        files = os.listdir(path)
        files.sort()  # order is important !
        files = np.reshape(files, (len(files) // 2, -1))
        images = files[:, 0]
        masks = files[:, 1]
        assert images.shape[0] == masks.shape[0]

        intended_size = math.floor(images.shape[0] / batch_size + 1)
        # Now the data will be partitioned according to the batch size
        images = np.array_split(images, intended_size, )

        masks = np.array_split(masks, intended_size)
        return images, masks

    def __extract_batch_data_array(self, file_names, is_training: bool):
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
