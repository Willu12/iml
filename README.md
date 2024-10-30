# iml

This project aims to build a Convolutional Neural Network (CNN) model for speech recognition using spectrograms generated from audio files. The model is trained to classify audio clips based on their spectrogram representations. The project is built with PyTorch and integrates Weights & Biases (W&B) for experiment tracking and performance logging.

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
│   ├── config.py                    # Configuration settings
│   ├── data_processing.py           # Spectrogram generation, augmentation, dataset split
│   ├── dataset_analysis.py          # Dataset exploration and visualization functions
│   ├── model.py                     # CNN model definition
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

## Data Preparation

1. **Download the Dataset**: Download the dataset from [Zenodo](https://zenodo.org/records/4660670) and place all `.wav` files in the `data/` directory.
2. **Process the Data**: Use the `data_preparation_and_analysis.ipynb` notebook to:
   - Load audio files
   - Split them into train, validation, and test sets
   - Divide each recording into 2-second clips
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

Spectrogram generation is handled by the `data_processing.py` script. Key functions include:
- `load_audio()`: Loads audio from a .wav file.
- `split_into_clips()`: Splits an audio file into 2-second clips.
- `create_spectrogram()`: Converts audio clips into log-mel spectrograms.

### Model Definition

The `SimpleCNN` model in `model.py` is a straightforward CNN classifier targeting an F1 score > 0.6. Customize the architecture in `model.py` to improve model performance as needed.

### Logging with Weights & Biases

Experiment metrics are logged to W&B:
- **Training Loss** and **Validation F1 Score** for each epoch.
- **Model Checkpoints** saved for tracking different training runs.

## Example Commands

```bash
# Activate the environment
source env/bin/activate

# Run Jupyter Notebook server
jupyter notebook
```

## Notes

- All modules in the `src/` directory are designed to be imported with a single alias, such as `import src as iml`.
- The project structure and code are optimized to allow easy loading and use in Jupyter notebooks for exploratory analysis.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Zenodo Dataset](https://zenodo.org/records/4660670): for providing the audio data used in this project.
- [Weights & Biases](https://wandb.ai/): for experiment tracking and logging.
