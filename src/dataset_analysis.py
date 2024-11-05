import matplotlib.pyplot as plt
import numpy as np
from .data_processing import load_audio, split_into_clips


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

def dataset_clips(file_list):
    """
    Returns list of track clips from .wav file list.
    
    Parameters:
        file_list (List[str]): List of .wav file paths.

    Returns:
        List[float64]: List of track durations
    """
    for file in file_list:
        audio, sr = load_audio(file)
        clips = split_into_clips(audio, sample_rate=sr)


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


def clips_statistics(clips, sr):
    return ClipsStatistics(clips, sr)

def statistics(data_files):
    """
    Returns statsitics of given list of files, such as total coun, mean and range.

    Parameters:
        file_list (List[str]): List of file paths in the dataset.

    Returns:
        DatasetStatistics
    """
    return DatasetStatistics(data_files)

class DatasetStatistics:
    def __init__(self, data_files):
        self.data_files = data_files
        self.durations = dataset_durations(self.data_files)
        self.get_stats_from_durations(self.durations);
        self.audios = [load_audio(file) for file in data_files]
        self.audio_signals = [audio[0] for audio in self.audios]

    def get_stats_from_durations(self, durations):
        self.durations = durations
        self.files_count = len(self.durations)
        self.total_duration = sum(self.durations)
        self.average_duration = np.mean(self.durations)
        self.duration_range = [min(self.durations),max(self.durations)]

    def __str__(self):
        return rf"""Statstics:
        Total files: {self.files_count},
        Total duration: {self.total_duration:.2f} sec,
        Average duration: {self.average_duration:.2f} sec, 
        Duration range: {self.duration_range[0]:.2f} - {self.duration_range[0]:.2f} sec
        """
    
class ClipsStatistics:
    def __init__(self, clips, sr):
        self.clips = clips;
        self.sr = sr;
        self.files_count = len(clips)
        durations = [len(clip) / sr for clip in clips]
        self.get_stats_from_durations(durations)          

    def get_stats_from_durations(self, durations):
        self.durations = durations
        self.files_count = len(self.durations)
        self.total_duration = sum(self.durations)
        self.average_duration = np.mean(self.durations)
        self.duration_range = [min(self.durations),max(self.durations)]
        
    
    def __str__(self):
        return rf"""Statstics:
        Total files: {self.files_count},
        Total duration: {self.total_duration:.2f} sec,
        Average duration: {self.average_duration:.2f} sec, 
        Duration range: {self.duration_range[0]:.2f} - {self.duration_range[0]:.2f} sec
        """