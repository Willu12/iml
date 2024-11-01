import matplotlib.pyplot as plt
import numpy as np
from .data_processing import load_audio


def plot_spectrogram(spectrogram, title="Spectrogram"):
    """
    Plots a single spectrogram.

    Parameters:
        spectrogram (np.ndarray): Spectrogram array to plot.
        title (str): Title for the plot.

    Returns:
        None
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect="auto", cmap="gray_r")
    plt.title(title)
    plt.colorbar(format="%+2.0f dB")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()


def dataset_durations(file_list):
    """
    Returns list of track durations from .wav file list.

    Parameters:
        file_list (List[str]): List of .wav file paths.

    Returns:
        List[float64]: List of track durations
    """
    durations = []
    for file in file_list:
        audio, sr = load_audio(file)
        durations.append(len(audio) / sr)
    return durations


def print_duration_summary(durations):
    """
    Prints a summary of sample duration statistics, such as total count, mean and range.

    Parameters:
        file_list (List[str]): List of file paths in the dataset.

    Returns:
        None
    """
    print(f"Total files: {len(durations)}")
    print(f"Total duration: {sum(durations):.2f} sec")
    print(f"Average duration: {np.mean(durations):.2f} sec")
    print(f"Duration range: {min(durations):.2f} - {max(durations):.2f} sec")
