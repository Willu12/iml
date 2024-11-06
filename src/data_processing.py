"""
Module for processing audio files, including loading, splitting into clips, 
spectrogram generation, and dataset preparation.
"""

import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from .config import (
    SAMPLE_RATE,
    TEST_DATASET_RATIO,
    VALIDATION_DATASET_RATIO,
    RANDOM_STATE,
)


def list_audio_files(data_dir):
    """
    Lists all supported files in a directory.
    Currently, it filters for '.wav' files that do not start with '._'.

    Parameters:
        data_dir (str): Directory containing .wav files.

    Returns:
        list[str]: List of paths to audio files.
    """
    return [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".wav") and not f.startswith("._")
    ]


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


def split_into_clips(audio, clip_duration=3, sr=SAMPLE_RATE):
    """
    Splits audio into fixed-duration clips.

    Parameters:
        audio (np.ndarray): Audio signal array.
        clip_duration (int): Duration of each clip in seconds.
        sr (int): Sampling rate of the audio.

    Returns:
        list[np.ndarray]: List of audio clips.
    """
    clip_length = clip_duration * sr
    return [
        audio[i : i + clip_length]
        for i in range(0, len(audio), clip_length)
        if len(audio[i : i + clip_length]) == clip_length
    ]


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


def create_spectrogram(audio, sr=SAMPLE_RATE, n_fft=1024, hop_length=512):
    """
    Converts an audio signal into a log-mel spectrogram.

    Parameters:
        audio (np.ndarray): Audio signal array.
        sr (int): Sampling rate of the audio.
        n_fft (int): Length of the FFT window.
        hop_length (int): Number of samples between successive frames.

    Returns:
        np.ndarray: Log-mel spectrogram of the audio.
    """
    spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return log_spectrogram


def prepare_datasets(
    data_dir, test_size=TEST_DATASET_RATIO, validation_size=VALIDATION_DATASET_RATIO
):
    """
    Splits audio files into training, validation, and test sets.

    Parameters:
        data_dir (str): Directory containing .wav files.
        test_size (float): Proportion of files to include in the test split.
        validation_size (float): Proportion of files to include in the validation split.

    Returns:
        tuple: Tuple containing lists of file paths for train, validation, and test sets.
    """
    files = list_audio_files(data_dir)
    train_files, test_files = train_test_split(
        files, test_size=test_size, random_state=RANDOM_STATE
    )

    val_ratio_local = validation_size / (1 - test_size)
    train_files, val_files = train_test_split(
        train_files, test_size=val_ratio_local, random_state=RANDOM_STATE
    )
    return train_files, val_files, test_files
