import source.config as config

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


class PreprocessImage:
    """
    Performs the necessary preprocessing for the image
    """
    def __init__(self, image_path: str, mask_path: str):
        self.image = sitk.ReadImage(image_path)
        self.mask = sitk.ReadImage(mask_path)

    def to_nparray(self):
        self.image = sitk.GetArrayFromImage(self.image)
        self.mask = sitk.GetArrayFromImage(self.mask)

    def normalize(self):
        self.image = self.__normalize_image(self.image)

    def standardize(self):
        mean = np.mean(self.image)
        std = np.std(self.image)
        standardized_image = (self.image - mean) / std
        self.image = standardized_image

    def resize(self):
        self.image = self.__resize_image(self.image)
        self.mask = self.__resize_image(self.mask)

    @staticmethod
    def __normalize_image(image):
        min = float(image.min())
        max = float(image.max())
        return  (image - min) / (max - min)

    def display_slice(self, index):
        plt.figure(figsize=[15, 8])
        image = self.image[index, :, :]
        mask = self.mask[index, :, :]

        plt.subplot(2, 1, 1)
        plt.title("Image")
        plt.imshow(image)

        plt.subplot(2, 1, 2)
        plt.title("Mask")
        plt.imshow(mask)

    @staticmethod
    def __resize_image(image: sitk.Image):
        image_size = image.GetSize()  # [x,y,z]
        new_size = config.image_size.copy()
        new_size.append(image_size[2])  # As we are only in need to resize the height and the width

        reference_image = sitk.Image(new_size, image.GetPixelIDValue())

        reference_image.SetOrigin(image.GetOrigin())
        reference_image.SetDirection(image.GetDirection())
        reference_image.SetSpacing([
            size * spacing / new_size
            for new_size, size, spacing in zip(new_size, image.GetSize(),
                                               image.GetSpacing())
        ])
        reference_image = sitk.Resample(image, reference_image)
        return reference_image
