import os

image_size = [128, 128]

#
val_set_ratio = 0.2
test_set_ratio = 0.2
# dataset paths
dataset_path = os.path.join(os.getcwd(), "data", "dataset")
masks_dataset_path = os.path.join(dataset_path, "masks")
images_dataset_path = os.path.join(dataset_path, "images")

# training, validation and test data paths
preprocessed_base_path = os.path.join(os.getcwd(), "data", "preprocessed_data")
training_set_path = os.path.join(preprocessed_base_path, "training_set")
test_set_path = os.path.join(preprocessed_base_path, "training_set")
validation_set_path = os.path.join(preprocessed_base_path, "training_set")

# number_of data
num_data = 100
