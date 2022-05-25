import os

import source.config as config
from source.trainer import SegNetTrainer

from tensorflow import keras
from tensorflow.keras.metrics import IoU
from tensorflow_addons.metrics import F1Score
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


class TestModel:
    """
    Performs the necessary operation for the model testing

    This include, displaying the predictions, displaying the model evolutions
    """

    def __init__(self, path_to_model):
        self.model = keras.models.load_model(path_to_model,
                                             custom_objects=None, compile=False, options=None
                                             )

    def predict(self, path_to_image, path_to_mask):
        """
        Receives an image and returns a plot comparing the prediction
        with the ground truth

        :param path_to_image: path to the image to predict
        :param path_to_mask: path to the mask to predict
        :return:
        """
        image_ = self.__get_image(path_to_image)  # image: [H,W]
        mask = self.__get_image(path_to_mask)

        image = np.expand_dims(image_, axis=0)  # image shape : [C,H,W]
        predicted = self.model.predict(image)  # shape:  [5,C,H,W] (5 are the number of the classes )

        predicted = np.argmax(predicted, axis=-1)
        cmap = plt.cm.viridis
        custom_lines = [Line2D([0], [0], color=cmap(1.), lw=4),
                        Line2D([0], [0], color=cmap(0.75), lw=4),
                        Line2D([0], [0], color=cmap(0.50), lw=4),
                        Line2D([0], [0], color=cmap(0.25), lw=4), ]

        plt.subplot(2, 1, 1)
        plt.title("Ground truth")
        plt.imshow(image_, cmap='gray')
        plt.imshow(mask, alpha=0.6, cmap='viridis')
        plt.legend(custom_lines, ['Pancreas', 'spleen', 'Kidney', 'liver'], bbox_to_anchor=(1.04, 1), loc="upper left")

        plt.subplot(2, 1, 2)
        plt.title("Predicted")
        plt.imshow(image_, cmap='gray')
        plt.imshow(predicted[0], alpha=0.6, cmap='viridis')
        plt.show()

    @staticmethod
    def __get_image(path):
        sitk_image = sitk.ReadImage(path)
        image_arr = sitk.GetArrayFromImage(sitk_image)
        return image_arr

    def compute_model_performance_perclass(self):
        path = config.test_set_path
        test_images, test_masks = SegNetTrainer.get_files(path, False)
        jaccard_indices = []
        counter = 0

        for f_image, f_mask in zip(test_images, test_masks):
            if counter == 10 : break
            image_path = os.path.join(path, f_image)
            mask_path = os.path.join(path, f_mask)

            image = self.__get_image(image_path)
            mask = self.__get_image(mask_path)
            mask_cate = keras.utils.to_categorical(mask, num_classes=config.num_classes, dtype='float32')

            image = np.expand_dims(image, axis=0)  # image shape : [C,H,W]
            predicted = self.model.predict(image)  # shape:  [5,C,H,W] (5 are the number of the classes )
            jaccard  = self.__compute_metric(predicted, mask_cate)

            jaccard_indices.append(jaccard)

            counter += 1

        jaccard_indices = np.array(jaccard_indices)
        self.__compute_mean_metric_per_class(jaccard_indices)

    @staticmethod
    def __compute_metric(prediction, ground_truth):
        iou = []
        for i in range(prediction.shape[3]):
            metric = IoU(2, target_class_ids=[0])
            metric.update_state(prediction[0, :, :, i], ground_truth[:, :, i])
            iou.append(metric.result().numpy())
        return iou

    @staticmethod
    def __compute_mean_metric_per_class(jaccard_metric):

        liver_mean_jaccard = np.mean(jaccard_metric[:, 1])
        kidney_mean_jaccard = np.mean(jaccard_metric[:, 2])
        spleen_mean_jaccard = np.mean(jaccard_metric[:, 3])
        pancreas_mean_jaccard = np.mean(jaccard_metric[:, 4])

        print("Jaccard Index :")
        print(f"The liver :{liver_mean_jaccard}")
        print(f"The kidney :{kidney_mean_jaccard}")
        print(f"The spleen :{spleen_mean_jaccard}")
        print(f"The pancreas :{pancreas_mean_jaccard}")

    @staticmethod
    def display_training_evolution(path_to_metrics):
        """
        Displays the model metrics during the training

        :param path_to_metrics: The csv dile containing all the metrics
        :return:
        """
        df = pd.read_csv(path_to_metrics)
        epoch = (df["epoch"]).tolist()
        train_loss = (df["train_loss"]).tolist()
        val_loss = (df["val_loss"]).tolist()
        accuracy = (df["accuracy"]).tolist()

        plt.figure()
        plt.plot(epoch, train_loss, label="Train loss")
        plt.plot(epoch, val_loss, label="Val loss")
        plt.plot(epoch, accuracy, label="Accuracy")
        plt.legend()
        plt.show()
