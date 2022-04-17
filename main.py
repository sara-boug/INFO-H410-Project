import argparse
import source.config as config
from source.generate_data import TrainingDataGenerator


class ModeTrainer:

    @staticmethod
    def generate_data():
        data_generator = TrainingDataGenerator(images_path=config.images_dataset_path,
                                               masks_path=config.masks_dataset_path)
        data_generator.execute()


if __name__ == "__main__":
    # Handle arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--preprocess', action='store_true', required=False)

    arguments = parser.parse_args()
    model_trainer = ModeTrainer()
    cuda_num = 0

    if arguments.preprocess:
        model_trainer.generate_data()
