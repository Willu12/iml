"""
Module providing helper functions and classes for analyzing audio data,
including spectrogram plotting and duration statistics.
"""

import matplotlib.pyplot as plt
import numpy as np
from .config import HOP_LENGTH, SAMPLE_RATE


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

def duration_statistics_spectrograms(spectograms, sr=SAMPLE_RATE, hop_length = HOP_LENGTH):
    """
    Computes duration statistics for a list of audio spectograms.

    Parameters:
        spectograms (np.ndarray): Log-mel spectrogram of the audio.
        sr (int): Sampling rate of the spectogram.
        hop_length (int) : Hop Length used to create spectogram.

    Returns:
        DurationStatistics: An instance containing calculated statistics.
    """
    duration_statistics =  DurationStatistics.__new__(DurationStatistics)
    duration_statistics.statistics_from_spectrograms(spectograms, sr, hop_length)
    return duration_statistics

def duration_statistics(clips, sr=SAMPLE_RATE):
    """
    Computes duration statistics for a list of audio clips.

    Parameters:
        clips (list[np.ndarray]): List of audio clip arrays.
        sr (int): Sampling rate of the audio clips.

    Returns:
        DurationStatistics: An instance containing calculated statistics.
    """
    return DurationStatistics(clips, sr=sr)


class DurationStatistics:
    """
    Class for computing and storing statistics about audio clip durations.

    Parameters:
        clips (list[np.ndarray]): List of audio clip arrays.
        sr (int): Sampling rate of the audio clips.

    Attributes:
        files_count (int): Total number of audio clips.
        total_duration (float): Sum of durations for all clips in seconds.
        average_duration (float): Average duration of clips in seconds.
        duration_range (list[float]): Minimum and maximum durations of clips in seconds.
    """
    
    def _update_from_durations(self, durations):
        self.files_count = len(self.durations)
        self.total_duration = sum(self.durations)
        self.average_duration = np.mean(self.durations)
        self.duration_range = [min(self.durations), max(self.durations)]

    def __init__(self, clips, sr=SAMPLE_RATE):
        self.durations = [len(clip) / sr for clip in clips]
        self._update_from_durations(self.durations)

    def statistics_from_spectrograms(self, spectrograms, sr = SAMPLE_RATE, hop_length = HOP_LENGTH):
        self.durations = [spectrogram.shape[1] * hop_length/sr for spectrogram in spectrograms]
        self._update_from_durations(self.durations)

    def __str__(self):
        return rf"""Statistics:
        Total files: {self.files_count},
        Total duration: {self.total_duration:.2f} sec,
        Average duration: {self.average_duration:.2f} sec, 
        Duration range: {self.duration_range[0]:.2f} - {self.duration_range[1]:.2f} sec
        """
