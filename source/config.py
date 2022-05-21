import os

image_size = [128, 128,]  # Height * Width* Channel

#
val_set_ratio = 0.2
test_set_ratio = 0.2
# dataset paths
dataset_path = os.path.join(os.getcwd(), "data", "dataset")
masks_dataset_path = os.path.join(dataset_path, "masks")
images_dataset_path = os.path.join(dataset_path, "images")

# training, validation and test data paths
preprocessed_base_path = os.path.join(os.getcwd(), "data", "preprocessed_data")
preprocessed_all = os.path.join(preprocessed_base_path, "all")

training_set_path = os.path.join(preprocessed_base_path, "training_set")
test_set_path = os.path.join(preprocessed_base_path, "test_set")
validation_set_path = os.path.join(preprocessed_base_path, "validation_set")

# model and metrics path

model_folder = os.path.join(os.getcwd(), "data", "generated_models")
model_path = os.path.join(model_folder, "segnet_model.h5")
metrics_path = os.path.join(model_folder, "metrics.csv")

# number_of data
num_data = 5  # reflect the original 3D data that has been used 
batch_size = 4
epochs = 1000
num_classes = 5
network_input_shape = (batch_size, 128, 128, 1)
