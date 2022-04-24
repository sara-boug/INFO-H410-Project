DATA="data"
P_DATA_FOLDER="preprocessed_data"

echo "Generating Data folder"

mkdir -p "$DATA"
mkdir -p "$DATA"/dataset
mkdir -p "$DATA"/generated_models
mkdir -p "$DATA"/loaded_data
mkdir -p "$DATA"/"$P_DATA_FOLDER"
mkdir -p "$DATA"/"$P_DATA_FOLDER"/test_set
mkdir -p "$DATA"/"$P_DATA_FOLDER"/training_set
mkdir -p "$DATA"/"$P_DATA_FOLDER"/validation_set


