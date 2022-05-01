import os

from source.preprocess_image import PreprocessImage
import source.config as config

import SimpleITK as sitk


class TrainingDataGenerator:
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path

    def execute(self):
        images_dir_files= os.listdir(self.images_path)
        images_dir_files = [file_name for file_name in images_dir_files if ".nii" in file_name ]
        images_dir_files.sort()

        masks_dir_files = os.listdir(self.masks_path) 
        masks_dir_files = [file_name for file_name in masks_dir_files if ".nii" in file_name ]
        masks_dir_files .sort()

        assert len(images_dir_files) == len(masks_dir_files)  # the masks and the images should be equal
        counter = 0
        counter_index = 0
        total = config.num_data
        dest_path = config.training_set_path
        for i in range(0, len(images_dir_files)):
            if 1 - counter / total <= config.test_set_ratio + config.val_set_ratio:
                dest_path = config.validation_set_path
            if 1 - counter / total <= config.test_set_ratio:
                dest_path = config.test_set_path

            image_path = os.path.join(self.images_path, images_dir_files[i])
            mask_path = os.path.join(self.masks_path, masks_dir_files[i])
            preprocess_image_obj = PreprocessImage(image_path=image_path, mask_path=mask_path)

            preprocess_image_obj.resize()
            preprocess_image_obj.to_nparray()
            preprocess_image_obj.standardize()
            preprocess_image_obj.normalize()

            image = preprocess_image_obj.image
            mask = preprocess_image_obj.mask
            assert image.shape[0] == mask.shape[0]
            # Save the slices
            for j in range(0, image.shape[0]):
                self.__write_image(
                    image=image[j, :, :],
                    mask=mask[j, :, :],
                    index=counter_index,
                    path=dest_path
                )
                counter_index += 1
            counter += 1

    def __write_image(self, image, mask, index, path):
        index_str = f"{index}"
        if index < 10:
            index_str = f"0{index}"
        dest_image_path = os.path.join(path, f"image{index_str}.nii")
        dest_mask_path = os.path.join(path, f"image{index_str}_segmentation.nii")

        sitk_image = sitk.GetImageFromArray(image)
        sitk_mask = sitk.GetImageFromArray(mask)

        sitk.WriteImage(sitk_image, dest_image_path)
        sitk.WriteImage(sitk_mask, dest_mask_path)
