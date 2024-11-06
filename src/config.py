""" Constant values consistent through the whole project """

SAMPLE_RATE = 16000

# Dataset split configuration
TEST_DATASET_RATIO = 0.2
VALIDATION_DATASET_RATIO = 0.16
RANDOM_STATE = 42

# Directory structure
DATA_DIR = "./data"  # Directory where .wav files are stored
DATASET_DIR = "./datasets"  # Directory to store processed spectrograms
TRAIN_DIR = f"{DATASET_DIR}/train/"
VAL_DIR = f"{DATASET_DIR}/val/"
TEST_DIR = f"{DATASET_DIR}/test/"

VALID_ACCESS_LABELS = {
    "f1",
    "f7",
    "f8",
    "m3",
    "m6",
    "m8",
}  # Speakers of such labels should be admitted to enter
