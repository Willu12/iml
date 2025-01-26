import os
import re
import random
from collections import defaultdict
from typing import List, Set
from .config import RANDOM_STATE

class AudioDatasetProcessor:
    def get_datasets(self):
        raise NotImplementedError()

    def compute_statistics(self):
        raise NotImplementedError()

class DAPSDatasetProcessor(AudioDatasetProcessor):
    def __init__(self, data_dir: str, valid_access_labels: Set[str], allowed_dictionaries: List[str]):
        self.data_dir = data_dir
        self.valid_access_labels = valid_access_labels
        self.allowed_dictionaries = allowed_dictionaries
        self.script_to_files = defaultdict(list)
        self.file_regex = re.compile(r'([fm]\d+)_script(\d+)_')
        self.scripts_count = 5
        self.training_scripts_count = 3
        self.validation_scripts_count = 1
        random.seed(RANDOM_STATE)

        self.dataset = {
            "train": [],
            "validate": [],
            "test": []
        }

        self._prepare_datasets()

    def get_datasets(self):
        return (self.dataset["train"], self.dataset["validate"], self.dataset["test"])

    def compute_statistics(self):
        datasets = self.dataset

        for name, dataset in datasets.items():
            total_samples = len(dataset)
            speakers = set()
            speaker_sample_counts = defaultdict(int)
            authorized_count = 0
            unauthorized_count = 0

            for filepath in dataset:
                filename = os.path.basename(filepath)
                match = self.file_regex.match(filename)
                if match:
                    speaker_tag = match.group(1)
                    speakers.add(speaker_tag)
                    speaker_sample_counts[speaker_tag] += 1
                    if speaker_tag in self.valid_access_labels:
                        authorized_count += 1
                    else:
                        unauthorized_count += 1

            print(f"--- {name.capitalize()} Set Statistics ---")
            print(f"Total Samples: {total_samples}")
            print(f"Total Speakers: {len(speakers)}")
            print(f"Authorized Samples: {authorized_count}")
            print(f"Unauthorized Samples: {unauthorized_count}")
            print(f"Authorized to Unauthorized Ratio: {authorized_count}:{unauthorized_count}")
            print("\nSamples per Speaker:")
            for speaker in sorted(speaker_sample_counts.keys()):
                print(f"  {speaker}: {speaker_sample_counts[speaker]}")
            print()

    def _prepare_datasets(self):
        print(f"Searching in allowed directories: {self.allowed_dictionaries}")
        wav_files_all = self._list_audio_files_recursively()
        print(f"Found {len(wav_files_all)} .wav files in directory '{self.data_dir}'")

        for filepath in wav_files_all:
            filename = os.path.basename(filepath)
            match = self.file_regex.match(filename)
            if match:
                script_number = int(match.group(2))
                if script_number <= self.training_scripts_count:
                    self.dataset["train"].append(filepath)
                elif script_number <= self.validation_scripts_count + self.training_scripts_count:
                    self.dataset["validate"].append(filepath)
                elif script_number <= self.scripts_count:
                    self.dataset["test"].append(filepath)
                else:
                    raise RuntimeError("Unexpected script number! It seems to not be DAPS dataset.")
            else:
                print(f"Filename {filename} does not match the expected pattern.")

    def _list_audio_files_recursively(self) -> List[str]:
        return [
            os.path.join(root, file)
            for root, _, files in os.walk(self.data_dir)
            for file in files
            if file.endswith('.wav') and any(d in root for d in self.allowed_dictionaries)
        ]