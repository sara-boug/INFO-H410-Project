import source.config as config

from tensorflow import keras
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


class TestModel:
    def __init__(self, path_to_model):
        self.model = keras.models.load_model(path_to_model,
                                             custom_objects=None, compile=False, options=None
                                             )

    def predict(self, path_to_image):
        image = self.__get_image(path_to_image)
        # image = np.expand_dims(image, axis=-1)

        image = np.expand_dims(image, axis=0)
        predicted = self.model.predict(image)

        predicted = np.argmax(predicted, axis=-1)
        plt.figure()
        plt.imshow((predicted)[0])
        plt.show()

        return predicted

    @staticmethod
    def __get_image(path):
        sitk_image = sitk.ReadImage(path)
        image_arr = sitk.GetArrayFromImage(sitk_image)
        return image_arr
