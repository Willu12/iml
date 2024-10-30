import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split

def load_audio(file_path, sample_rate=16000):
    """
    Loads an audio file.

    Parameters:
        file_path (str): Path to the .wav file.
        sr (int): Sampling rate to use when loading audio.

    Returns:
        np.ndarray: Loaded audio signal.
        int: Sampling rate.
    """
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio, sr

def create_spectrogram(audio, sample_rate, n_fft=1024, hop_length=512):
    """
    Converts an audio clip into a log-mel spectrogram.

    Parameters:
        audio_clip (np.ndarray): Audio clip data.
        sr (int): Sampling rate of the audio clip.
        n_mels (int): Number of mel bands to generate.

    Returns:
        np.ndarray: Log-mel spectrogram of the audio clip.
    """
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_fft=n_fft, hop_length=hop_length)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    return log_spectrogram

def split_into_clips(audio, clip_duration=2, sample_rate=16000):
    """
    Splits audio into fixed-duration clips.

    Parameters:
        audio (np.ndarray): Audio signal array.
        sr (int): Sampling rate of the audio.
        clip_duration (int): Duration (seconds) of each clip.

    Returns:
        List[np.ndarray]: List of audio clips.
    """
    clip_length = clip_duration * sample_rate
    return [audio[i:i + clip_length] for i in range(0, len(audio), clip_length) if len(audio[i:i + clip_length]) == clip_length]

def augment_audio(audio):
    """
    Applies basic audio augmentations to an audio clip.

    Parameters:
        audio_clip (np.ndarray): Audio data for augmentation.
        augmentation_type (str): Type of augmentation ("noise" or "shift").
        noise_factor (float): Noise factor when adding noise.

    Returns:
        np.ndarray: Augmented audio clip.
    """
    audio = librosa.effects.pitch_shift(audio, sr=16000, n_steps=np.random.uniform(-1, 1))
    noise = np.random.normal(0, 0.005, audio.shape)
    return audio + noise

def prepare_datasets(data_dir):
    """
    Splits audio files into training, validation, and test sets.

    Parameters:
        data_dir (str): Directory containing .wav files.
        test_size (float): Proportion of data to use as the test set.
        val_size (float): Proportion of data to use as the validation set.

    Returns:
        Tuple[List[str], List[str], List[str]]: Paths for train, validation, and test sets.
    """
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]
    train_files, test_files = train_test_split(files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.2, random_state=42)
    return train_files, val_files, test_files
