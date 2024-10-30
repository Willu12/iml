DATA_DIR = "./data"          # Directory where .wav files are stored
DATASET_DIR = "./datasets"    # Directory to store processed spectrograms
TRAIN_DIR = f"{DATASET_DIR}/train/"
VAL_DIR = f"{DATASET_DIR}/val/"
TEST_DIR = f"{DATASET_DIR}/test/"
# Speakers of such labels should be admitted to enter
VALID_ACCESS_LABELS = {
    "f1", "f7", "f8", "m3", "m6", "m8"
}