import os
import SimpleITK as sitk

from source.preprocess_image import PreprocessImage
import source.base_parameters as params


class GenerateData:
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path

    def generate(self):
        images_dir_files = os.listdir(self.images_path)
        masks_dir_files = os.listdir(self.masks_path)
        assert len(images_dir_files) == len(masks_dir_files)  # the masks and the images should be equal
        counter = 0
        dest_path = params.training_set_path
        for i in range(0, len(images_dir_files)):
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
        dest_image_path = os.path.join(path, f"image_{index}.jpg")
        dest_mask_path = os.path.join(path, f"image_segmentation_{index}.jpg")

        sitk_image = sitk.GetImageFromArray(image)
        sitk_mask = sitk.GetImageFromArray(mask)

        sitk.WriteImage(sitk_image, dest_image_path)
        sitk.WriteImage(sitk_mask, dest_mask_path)
