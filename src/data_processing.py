"""
Module for processing audio files, including loading, splitting into clips,
spectrogram generation, and dataset preparation.
"""

import os
import json
from matplotlib import pyplot as plt
import numpy as np
import webrtcvad
import librosa
from PIL import Image
from sklearn.model_selection import train_test_split
from .config import (
    SAMPLE_RATE,
    TEST_DATASET_RATIO,
    VALIDATION_DATASET_RATIO,
    RANDOM_STATE,
)

from collections import Counter, defaultdict
import random

def list_all_audio_files(data_dir):
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

def list_audio_files_recursively(data_dir, allowed_dirs=None):
    """
    Lists all supported '.wav' files in a directory and its subdirectories.
    Optionally filters the directories to consider only specific ones.

    Parameters:
        data_dir (str): Directory containing .wav files.
        allowed_dirs (list[str], optional): List of directory names to filter. Defaults to None.

    Returns:
        list[str]: List of paths to audio files.
    """
    audio_files = []
    for root, dirs, files in os.walk(data_dir):
        # If allowed_dirs is provided, filter the directories
        if allowed_dirs is not None:
            dirs[:] = [d for d in dirs if d in allowed_dirs]

        for f in files:
            if f.endswith(".wav") and not f.startswith("._"):
                audio_files.append(os.path.join(root, f))
    return audio_files

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

def is_speech(clip, sr, vad):
    """
    Checks if the majority of the audio clip contains speech using WebRTC VAD.

    Parameters:
        clip (np.ndarray): Audio signal array.
        sr (int): Sampling rate of the audio.
        vad (webrtcvad.Vad): WebRTC VAD instance.

    Returns:
        bool: True if the clip contains significant speech, False otherwise.
    """
    clip = (clip * 32768).astype(np.int16)  # Convert to 16-bit PCM
    frame_duration_ms = 30  # WebRTC VAD works with 10, 20, or 30 ms frames
    frame_size = int(sr * frame_duration_ms / 1000)
    frames = [
        clip[i : i + frame_size] for i in range(0, len(clip), frame_size)
        if len(clip[i : i + frame_size]) == frame_size
    ]
    
    speech_frames = sum(vad.is_speech(frame.tobytes(), sr) for frame in frames)
    return speech_frames / len(frames) > 0.5  # More than 50% speech

def split_into_clips_with_speech_filter(audio, clip_duration=3, sr=16000):
    """
    Splits audio into fixed-duration clips and filters by speech activity.

    Parameters:
        audio (np.ndarray): Audio signal array.
        clip_duration (int): Duration of each clip in seconds.
        sr (int): Sampling rate of the audio.

    Returns:
        list[np.ndarray]: List of audio clips with significant speech.
    """
    vad = webrtcvad.Vad()
    vad.set_mode(2)  # Aggressiveness level: 0 (low) to 3 (high)
    
    clip_length = clip_duration * sr
    clips = [
        audio[i : i + clip_length]
        for i in range(0, len(audio), clip_length)
        if len(audio[i : i + clip_length]) == clip_length
    ]
    
    return [clip for clip in clips if is_speech(clip, sr, vad)]

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

def list_balanced_audio_files(data_dir, max_files_per_user=3):
    """
    Lists all supported files in a directory and balances the recordings per user.
    """
    audio_files = list_all_audio_files(data_dir)
    balanced_files = balance_recordings_by_user(audio_files, max_files_per_user)
    return balanced_files


def prepare_datasets(
    data_dir, listing_data_func=list_all_audio_files, test_size=TEST_DATASET_RATIO, validation_size=VALIDATION_DATASET_RATIO
):
    """
    Splits audio files into training, validation, and test sets.

    Parameters:
        data_dir (str): Directory containing audio files.
        listing_data_func (func): Function to list audio files in a directory.
        test_size (float): Proportion of files to include in the test split.
        validation_size (float): Proportion of files to include in the validation split.

    Returns:
        tuple: Tuple containing lists of file paths for train, validation, and test sets.
    """
    files = listing_data_func(data_dir)
    train_files, test_files = train_test_split(
        files, test_size=test_size, random_state=RANDOM_STATE
    )

    val_ratio_local = validation_size / (1 - test_size)
    train_files, val_files = train_test_split(
        train_files, test_size=val_ratio_local, random_state=RANDOM_STATE
    )
    return train_files, val_files, test_files


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

def extract_metadata(wav_files, valid_access_labels):
    """
    Extract metadata from a list of .wav file paths.

    Args:
        wav_files (list of str): List of paths to .wav files.
        valid_access_labels (set): Set of speakers with valid access.

    Returns:
        list of dict: List of metadata dictionaries for each .wav file.
    """
    metadata = []
    
    for path in wav_files:
        filename = os.path.basename(path).replace(".wav", "")
        parts = filename.split("_")
        if len(parts) < 4:
            raise ValueError(f"Filename '{filename}' does not contain enough parts for parsing.")
        
        speaker, script, device, room = parts[0], parts[1], parts[2], parts[3]
        metadata.append({
            "path": path,
            "speaker": speaker,
            "script": script,
            "device": device,
            "room": room,
            "authorized": speaker in valid_access_labels
        })
    
    return metadata

def exclude_overlapping_scripts(train, validation, test):
    # Collect training set scripts and devices
    train_combinations = set((file["script"], file["device"]) for file in train)

    # Exclude overlapping combinations in validation and test sets
    validation = [file for file in validation if (file["script"], file["device"]) not in train_combinations]
    test = [file for file in test if (file["script"], file["device"]) not in train_combinations]

    # Ensure validation and test sets have data by re-sampling from available files if empty
    if not validation:
        validation = random.sample(train, min(10, len(train)))
    if not test:
        test = random.sample(train, min(10, len(train)))

    return validation, test

def compute_statistics(files):
    speakers = Counter(file["speaker"] for file in files)
    devices = Counter(file["device"] for file in files)
    rooms = Counter(file["room"] for file in files)
    authorized_count = sum(1 for file in files if file["authorized"])
    unauthorized_count = len(files) - authorized_count

    return {
        "Total Files": len(files),
        "Authorized": authorized_count,
        "Unauthorized": unauthorized_count,
        "Speakers": speakers,
        "Devices": devices,
        "Rooms": rooms,
    }
    
def display_dataset_statistics(train_files, validation_files, test_files):
    """
    Computes and displays statistics for training, validation, and test datasets.

    Parameters:
        train_files (list): List of files in the training dataset.
        validation_files (list): List of files in the validation dataset.
        test_files (list): List of files in the test dataset.

    Returns:
        None
    """
    datasets = {
        "Train Dataset": train_files,
        "Validation Dataset": validation_files,
        "Test Dataset": test_files,
    }
    
    for dataset_name, files in datasets.items():
        stats = compute_statistics(files)
        print(f"{dataset_name} Statistics:", stats)

def save_spectrogram(spectrogram, output_path):
    plt.imsave(output_path, spectrogram, cmap='gray')
