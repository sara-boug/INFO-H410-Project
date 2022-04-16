import argparse
import source.base_parameters as params
from source.generate_data import GenerateData


class ModeTrainer:

    @staticmethod
    def generate_data():
        data_generator = GenerateData(images_path=params.images_dataset_path.masks,
                                      masks_path=params.masks_dataset_path)
        data_generator.generate()


if __name__ == "__main__":
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true', required=False)

    arguments = parser.parse_args()
    model_trainer = ModeTrainer()
    cuda_num = 0

    if arguments.preprocess:
        model_trainer.generate_data()
