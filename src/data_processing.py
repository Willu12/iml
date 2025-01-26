"""
Module for processing audio files, including loading, splitting into clips,
spectrogram generation, and dataset preparation.
"""

import os
import json
import numpy as np
import librosa
from PIL import Image
from .config import (
    SAMPLE_RATE,
)

def load_audio(file_path, sr=SAMPLE_RATE):
    """
    Loads an audio file.

    Parameters:
        file_path (str): Path to the .wav file.
        sr (int): Sampling rate to use when loading the audio.

    Returns:
        np.ndarray: Loaded audio signal.
    """
    audio, sr = librosa.load(file_path, sr=sr)
    return audio


def load_dir_audios(file_paths, sr=SAMPLE_RATE):
    """
    Loads multiple audio files.

    Parameters:
        file_paths (list[str]): List of paths to .wav files.
        sr (int): Sampling rate to use when loading audio.

    Returns:
        list[np.ndarray]: List of loaded audio signals.
    """
    return [load_audio(file_path, sr=sr) for file_path in file_paths]

class SOAAudioClips:
    """
    Iterator class for loading and iterating through audio clips from files.

    Parameters:
        file_paths (list[str]): List of paths to .wav files.
        sr (int): Sampling rate to use when loading audio.

    Attributes:
        file_paths (list[str]): List of paths to .wav files.
        clips (list[np.ndarray]): List of loaded audio files.

    Yields:
        tuple: Tuple containing file path and corresponding audio clip.
    """

    def __init__(self, file_paths, sr=SAMPLE_RATE):
        self.sr = sr
        self.file_paths = file_paths
        self.clips = load_dir_audios(self.file_paths, sr=sr)

    def __len__(self):
        return len(self.file_paths)

    def __iter__(self):
        for file_path, clip in zip(self.file_paths, self.clips):
            yield file_path, clip

def load_gray_image_normalized(filepath):
    """
    Loads an image file, converts it to grayscale, and normalizes pixel values.

    Parameters:
        filepath (str): Path to the image file.

    Returns:
        np.ndarray: Normalized grayscale image as a 2D array with values between 0 and 1.
    """
    image = Image.open(filepath).convert("L")  # Convert to grayscale

    # Convert image to a NumPy array and normalize to [0, 1]
    return np.array(image, dtype=np.float32) / 255.0

def compute_mean_std_from_images(directory):
    """
    Computes the mean and standard deviation of pixel values across all grayscale images in a directory.

    Parameters:
        directory (str): Directory containing image files.

    Returns:
        tuple[float, float]: Tuple of overall mean and standard deviation of pixel values.
    """
    means, stds = [], []

    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image_array = load_gray_image_normalized(image_path)

            means.append(np.mean(image_array))
            stds.append(np.std(image_array))

    overall_mean = np.mean(means)
    overall_std = np.mean(stds)

    return overall_mean, overall_std


def save_mean_std(mean, std, file_path):
    """
    Saves mean and standard deviation to a JSON file.

    Parameters:
        mean (float): Mean value to save.
        std (float): Standard deviation value to save.
        file_path (str): Path to the JSON file.
    """
    data = {"mean": float(mean), "std": float(std)}
    with open(file_path, "w") as f:
        json.dump(data, f)


def load_mean_std(file_path):
    """
    Loads mean and standard deviation from a JSON file.

    Parameters:
        file_path (str): Path to the JSON file.

    Returns:
        tuple: A tuple containing mean and std.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    mean, std = data["mean"], data["std"]
    return mean, std
