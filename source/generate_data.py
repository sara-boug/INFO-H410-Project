import os

from source.preprocess_image import PreprocessImage
import source.config as config

import SimpleITK as sitk
import numpy as np
from sklearn.model_selection import train_test_split


class TrainingDataGenerator:
    """
    Contains the necessary functionalities to Generate the data set for the whole pipline
    """
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path

    def execute(self):
        """
        Preprocess all the data and write them to a file

        The process is as follow :
            1 - The images and the masks are extracted from the folder (3D images)
            2 - The images are preprocessed
            3 - The slices are extracted
        :return:
        """
        # Extract the images and masks
        images_dir_files = os.listdir(self.images_path)
        images_dir_files = [file_name for file_name in images_dir_files if ".nii" in file_name]
        images_dir_files.sort()

        masks_dir_files = os.listdir(self.masks_path)
        masks_dir_files = [file_name for file_name in masks_dir_files if ".nii" in file_name]
        masks_dir_files.sort()
        assert len(images_dir_files) == len(masks_dir_files)  # the masks and the images should be equal
        # Counter for file indexing
        counter = 0
        counter_index = 0
        total = config.num_data
        # The path where all the dataset is located
        dest_path = config.preprocessed_all
        for i in range(0, len(images_dir_files)):
            image_path = os.path.join(self.images_path, images_dir_files[i])
            mask_path = os.path.join(self.masks_path, masks_dir_files[i])
            preprocess_image_obj = PreprocessImage(image_path=image_path, mask_path=mask_path)

            preprocess_image_obj.resize()
            preprocess_image_obj.to_nparray()

            # preprocess_image_obj.standardize()
            preprocess_image_obj.normalize()

            image = preprocess_image_obj.image
            mask = preprocess_image_obj.mask
            assert image.shape[0] == mask.shape[0]
            # Extract and save the slices
            for j in range(0, image.shape[0]):
                self.__write_image(
                    image=image[j, :, :],
                    mask=mask[j, :, :],
                    index=counter_index,
                    path=dest_path
                )
                counter_index += 1
            counter += 1

    def generate_files_per_set(self):
        """
        Extract the preprocessed slices and save them into 3 folders ( training, testing and validation)
        :return:
        """
        path = config.preprocessed_all
        files = os.listdir(path)
        files = np.reshape(files, (len(files) // 2, -1))
        mock_files = np.arange(0, files.shape[0])
        # Split the data into train and test
        train_set, test_set = train_test_split(mock_files, test_size=0.2)
        train_set, validation_set = train_test_split(train_set, test_size=0.2)
        read_path = config.preprocessed_all

        self.__write_file_set(read_path, config.test_set_path, files, test_set)
        self.__write_file_set(read_path, config.validation_set_path, files, validation_set)
        self.__write_file_set(read_path, config.training_set_path, files, train_set)

    @staticmethod
    def __write_file_set(read_path, write_path, files, set_indices):
        """
         Read the image from specific location and write it into the adequate set

        :param read_path: The folder to read the files from
        :param write_path: The folder to write the file into
        :param files: all the files
        :param set_indices: the test, train and val indices
        :return:
        """
        for index in set_indices:
            element = files[index]

            source_img_path = os.path.join(read_path, element[0])
            source_mask_path = os.path.join(read_path, element[1])

            dest_img_path = os.path.join(write_path, element[0])
            dest_mask_path = os.path.join(write_path, element[1])

            sitk_image = sitk.ReadImage(source_img_path)
            sitk_mask = sitk.ReadImage(source_mask_path)

            sitk.WriteImage(sitk_image, dest_img_path)
            sitk.WriteImage(sitk_mask, dest_mask_path)

    def __write_image(self, image, mask, index, path):
        if (mask > 3).sum() == 0:  # For the sake of simplicity, only images with organs are kept
            return

        index_str = f"{index}"
        if index < 10:
            index_str = f"0{index}"
        dest_image_path = os.path.join(path, f"image{index_str}.nii")
        dest_mask_path = os.path.join(path, f"image{index_str}_segmentation.nii")

        sitk_image = sitk.GetImageFromArray(image)
        sitk_mask = sitk.GetImageFromArray(mask)

        sitk.WriteImage(sitk_image, dest_image_path)
        sitk.WriteImage(sitk_mask, dest_mask_path)
