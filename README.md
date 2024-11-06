# iml

The goal of the project is to prepare a machine learning module that can be hypothetically
used in an automated, voice-based intercom device. Imagine that you are working in
a team of several programmers and the access to your floor is restricted with doors. There
is an intercom that can be used to open the door. You are implementing a machine
learning module that will recognize if a given person has the permission to open the door
or not. To simplify this problem, we interpret this as a binary recognition problem. Class
1 is made of allowed persons. Class 0 is made of not allowed persons. The medium used
to distinguish between the two classes is voice. Thus, the project to be delivered in in fact
a voice recognition project. To shift the focus to convolutional neural networks, we
assume that the voice recording must be turned into spectrograms in the pre-processing
stage. Thus, the project to be delivered is in fact a voice recognition project but
implementing it will require techniques that can also be applied to diMerent kinds of
image processing.
~ from Introduction to Machine Learning WUT course project description.

## Project Structure

```plaintext
iml/
│
├── data/                            # Raw .wav files directory
├── datasets/                        # Processed dataset directory with train, val, test spectrograms
├── notebooks/                       # Jupyter notebooks for data preparation and training
│   ├── data_preparation_and_analysis.ipynb   # Data processing and analysis notebook
│   ├── model_training.ipynb                  # Model training and logging notebook
├── src/                             # Main source files
│   ├── __init__.py
│   ├── data_processing.py           # Spectrogram generation, augmentation, dataset split
│   ├── dataset_analysis.py          # Dataset exploration and visualization functions
│   ├── model.py                     # CNN models definitions
│   └── training.py                  # Training and validation functions
├── models/                          # Saved model directory
├── requirements.txt                 # Python dependencies
└── README.md                        # Project description and setup guide
```

## Requirements

- Python 3.11
- Install project dependencies from `requirements.txt`:
  ```bash
  python3 -m venv env
  source env/bin/activate
  pip install -r requirements.txt
  ```
  or VSCode can execute this command automagically by using "Python: Create Enviroment..." action.
  To use jupyter notebooks (recommended way), `jupyter` package must be installed.

## Data Preparation

1. **Download the Dataset**: Download the dataset from [Zenodo](https://zenodo.org/records/4660670) and place chosen `.wav` files (I was using "clean" directory). Be aware of strange files starting with `.`, which are not proper `.wav` and needs to be removed.
2. **Process the Data**: Use the `data_preparation_and_analysis.ipynb` notebook to:
   - Load audio files
   - Split them into train, validation, and test sets
   - Divide each recording into 3-second clips
   - Generate spectrograms and save them to `datasets/` subdirectories
   - Generate dataset statistics and visualize sample spectrograms

## Model Training

1. **Initialize W&B**: Set up a Weights & Biases (W&B) account to log experiment metrics and track model performance.
2. **Run Model Training**: Use the `model_training.ipynb` notebook to:
   - Register a new W&B project
   - Load spectrograms for each dataset split
   - Train the CNN model and log performance metrics like training loss and validation F1 score
   - Save the trained model to `models/`

## Usage

### Spectrogram Generation

Spectrogram generation is handled by the `data_processing.py` functions:
- `load_audio()`: Loads audio from a .wav file.
- `split_into_clips()`: Splits an audio file into 3-second clips.
- `create_spectrogram()`: Converts audio clips into log-mel spectrograms.

### Model Definition

The `SimpleCNN` is simple CNN for 128x128 image size.
The `TutorialCNN` is the CNN from provided [tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).

## Acknowledgments

- [Zenodo Dataset](https://zenodo.org/records/4660670): for providing the audio data used in this project.
- [Weights & Biases](https://wandb.ai/): for experiment tracking and logging.
