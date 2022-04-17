import os

from source.preprocess_image import PreprocessImage
import source.config as config

import SimpleITK as sitk


class TrainingDataGenerator:
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path

    def execute(self):
        images_dir_files = os.listdir(self.images_path)
        masks_dir_files = os.listdir(self.masks_path)
        assert len(images_dir_files) == len(masks_dir_files)  # the masks and the images should be equal
        counter = 0
        total = config.num_data
        dest_path = config.training_set_path
        for i in range(0, len(images_dir_files)):
            if 1 - counter / total <= config.test_set_ratio + config.val_set_ratio:
                dest_path = config.val_set_ratio
            if 1 - counter / total <= config.test_set_ratio:
                dest_path = config.test_set_ratio

            image_path = os.path.join(self.images_path, images_dir_files[i])
            mask_path = os.path.join(self.masks_path, masks_dir_files[i])

            preprocess_image_obj = PreprocessImage(image_path=image_path, mask_path=mask_path)

            preprocess_image_obj.resize()
            preprocess_image_obj.to_nparray()
            preprocess_image_obj.standardize()
            preprocess_image_obj.normalize()

            image = preprocess_image_obj.image
            mask = preprocess_image_obj.mask
            # Save the slices
            for j in range(0, image.shape[0]):
                self.__write_image(
                    image=image[i, :, :],
                    mask=image[i, :, :],
                    index=counter,
                    path=dest_path
                )
            counter += 1

    def __write_image(self, image, mask, index, path):
        index_str = f"{index}"
        if index < 10:
            index_str = f"0{index}"
        dest_image_path = os.path.join(path, f"image_{index}.nii")
        dest_mask_path = os.path.join(path, f"image_segmentation_{index}.nii")

        sitk_image = sitk.GetImageFromArray(image)
        sitk_mask = sitk.GetImageFromArray(mask)

        sitk.WriteImage(sitk_image, dest_image_path)
        sitk.WriteImage(sitk_mask, dest_mask_path)
