import source.config as config

from tensorflow import keras
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

class TestModel:
    def __init__(self, path_to_model):
        self.model = keras.models.load_model(path_to_model,
                                             custom_objects=None, compile=False, options=None
                                             )

    def predict(self, path_to_image, path_to_mask):
        image_ = self.__get_image(path_to_image)
        mask = self.__get_image(path_to_mask)

        image = np.expand_dims(image_, axis=0)
        predicted = self.model.predict(image)

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

    @staticmethod
    def display_training_evolution(path_to_metrics):
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








