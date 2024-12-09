import random
import numpy as np
import librosa
import librosa.display
import webrtcvad
import matplotlib.pyplot as plt
from .config import SAMPLE_RATE, RANDOM_STATE

class AudioProcessor:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        random.seed(RANDOM_STATE)

    def load_audio(self, file_path):
        """Loads an audio file."""
        audio, _ = librosa.load(file_path, sr=self.sample_rate)
        return audio

    def split_audio(self, audio, clip_duration, factor = 1.5):
        """Splits audio into fixed-duration clips."""
        clip_length = int(clip_duration * self.sample_rate)
        num_segments = int((len(audio) / clip_length) * factor)
        return [self.crop_audio(audio, clip_duration) for _ in range(num_segments)]
    
    def crop_audio(self, audio, crop_duration):
        """Randomly crops an audio segment of specified duration."""
        crop_length = int(crop_duration * self.sample_rate)
        if len(audio) <= crop_length:
            return audio  # If audio is shorter than the crop, return as is.
        start = random.randint(0, len(audio) - crop_length)
        return audio[start:start + crop_length]

    def filter_speech_clips(self, audio_clips):
        """Filters clips to retain only those with significant speech."""
        vad = webrtcvad.Vad()
        vad.set_mode(2)  # Moderate aggressiveness
        return [clip for clip in audio_clips if self._contains_speech(clip, vad)]

    def _contains_speech(self, clip, vad):
        """Checks if a clip contains significant speech."""
        pcm_clip = (clip * 32768).astype(np.int16)  # Convert to 16-bit PCM
        frame_size = int(self.sample_rate * 0.03)  # 30 ms frames
        frames = [
            pcm_clip[i:i + frame_size]
            for i in range(0, len(pcm_clip), frame_size)
            if len(pcm_clip[i:i + frame_size]) == frame_size
        ]
        speech_frames = sum(
            vad.is_speech(frame.tobytes(), self.sample_rate) for frame in frames
        )
        return speech_frames / len(frames) > 0.5

    def create_spectrogram(self, audio):
        """Creates a log-mel spectrogram."""
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate)
        return librosa.power_to_db(mel_spec, ref=np.max)

    def save_spectrogram(self, spectrogram, output_path):
        """Saves the spectrogram as an image."""
        plt.imsave(output_path, spectrogram, cmap='gray')
