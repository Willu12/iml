import os
import re
import random
from collections import defaultdict
from typing import List, Set, Dict, Tuple
from .config import TRAIN_DATASET_RATIO, VALIDATION_DATASET_RATIO, RANDOM_STATE

class AudioDatasetProcessor:
    def __init__(self, data_dir: str, valid_access_labels: Set[str], allowed_dictionaries: List[str], train_ratio: float = TRAIN_DATASET_RATIO, validate_ratio: float = VALIDATION_DATASET_RATIO):
        self.data_dir = data_dir
        self.valid_access_labels = valid_access_labels
        self.allowed_dictionaries = allowed_dictionaries
        self.train_ratio = train_ratio
        self.validate_ratio = validate_ratio
        self.speaker_script_to_files = defaultdict(list)

        self.datasets = {
            "unbalanced": {
                "train": [],
                "validate": [],
                "test": []
            },
            "balanced": {
                "train": [],
                "validate": [],
                "test": []
            }
        }

        self._prepare_datasets()

    def get_datasets(self, balanced: bool = True):
        dataset_type = "balanced" if balanced else "unbalanced"
        return (self.datasets[dataset_type]["train"], self.datasets[dataset_type]["validate"], self.datasets[dataset_type]["test"])

    def list_audio_files_recursively(self) -> List[str]:
        return [
            os.path.join(root, file)
            for root, _, files in os.walk(self.data_dir)
            for file in files
            if file.endswith('.wav') and any(d in root for d in self.allowed_dictionaries)
        ]

    def parse_files(self):
        print(f"Searching in allowed directories: {self.allowed_dictionaries}")
        wav_files_all = self.list_audio_files_recursively()
        print(f"Found {len(wav_files_all)} .wav files in directory '{self.data_dir}'")

        pattern = re.compile(r'([fm]\d+)_script(\d+)_')
        for filepath in wav_files_all:
            filename = os.path.basename(filepath)
            match = pattern.match(filename)
            if match:
                speaker_tag = match.group(1)
                script_number = int(match.group(2))
                self.speaker_script_to_files[(speaker_tag, script_number)].append(filepath)
            else:
                print(f"Filename {filename} does not match the expected pattern.")

    def _prepare_datasets(self):
        self.parse_files()
        self.datasets["unbalanced"] = self._split_datasets(balance=False)
        self.datasets["balanced"] = self._split_datasets(balance=True)

    def _split_datasets(self, balance: bool) -> Dict[str, List[str]]:
        dataset_splits = {
            "train": [],
            "validate": [],
            "test": []
        }
        authorized_train_samples = []
        unauthorized_train_samples = []

        all_speakers = {speaker for speaker, _ in self.speaker_script_to_files.keys()}

        random.seed(RANDOM_STATE) # For reproducibility
        for speaker in all_speakers:
            speaker_scripts = [script for (spk, script) in self.speaker_script_to_files.keys() if spk == speaker]
            random.shuffle(speaker_scripts)

            num_scripts = len(speaker_scripts)
            splits = self._calculate_split_sizes(num_scripts)
            train_scripts, validate_scripts, test_scripts = self._assign_scripts_to_splits(speaker_scripts, splits)

            for script_list, split in zip([train_scripts, validate_scripts, test_scripts], ["train", "validate", "test"]):
                for script in script_list:
                    files = self.speaker_script_to_files[(speaker, script)]
                    dataset_splits[split].extend(files)

                    if split == "train":
                        if speaker in self.valid_access_labels:
                            authorized_train_samples.extend(files)
                        else:
                            unauthorized_train_samples.extend(files)

        if balance:
            self._balance_train_samples(dataset_splits, authorized_train_samples, unauthorized_train_samples)

        return dataset_splits

    def _calculate_split_sizes(self, num_scripts: int) -> Tuple[int, int, int]:
        num_train = max(1, int(self.train_ratio * num_scripts))
        num_validate = max(1, int(self.validate_ratio * num_scripts))
        num_test = num_scripts - num_train - num_validate

        if num_test == 0:
            num_test = 1
            num_train -= 1

        return num_train, num_validate, num_test

    def _assign_scripts_to_splits(self, scripts: List[int], splits: Tuple[int, int, int]) -> Tuple[List[int], List[int], List[int]]:
        num_train, num_validate, num_test = splits
        return (
            scripts[:num_train],
            scripts[num_train:num_train + num_validate],
            scripts[num_train + num_validate:]
        )

    def _balance_train_samples(self, dataset_splits: Dict[str, List[str]], authorized: List[str], unauthorized: List[str]):
        num_authorized = len(authorized)
        num_unauthorized = len(unauthorized)

        if num_authorized < num_unauthorized:
            random.shuffle(unauthorized)
            unauthorized = unauthorized[:num_authorized]
        else:
            random.shuffle(authorized)
            authorized = authorized[:num_unauthorized]

        dataset_splits["train"] = authorized + unauthorized

    def compute_statistics(self, balanced: bool = True):
        dataset_type = "balanced" if balanced else "unbalanced"
        datasets = self.datasets[dataset_type]

        pattern = re.compile(r'([fm]\d+)_script(\d+)_')

        for name, dataset in datasets.items():
            total_samples = len(dataset)
            speakers = set()
            authorized_count = 0
            unauthorized_count = 0
            speaker_sample_counts = defaultdict(int)

            for filepath in dataset:
                filename = os.path.basename(filepath)
                match = pattern.match(filename)
                if match:
                    speaker_tag = match.group(1)
                    speakers.add(speaker_tag)
                    speaker_sample_counts[speaker_tag] += 1
                    if speaker_tag in self.valid_access_labels:
                        authorized_count += 1
                    else:
                        unauthorized_count += 1

            print(f"--- {name.capitalize()} Set Statistics ({dataset_type.capitalize()}) ---")
            print(f"Total Samples: {total_samples}")
            print(f"Total Speakers: {len(speakers)}")
            print(f"Authorized Samples: {authorized_count}")
            print(f"Unauthorized Samples: {unauthorized_count}")
            print(f"Authorized to Unauthorized Ratio: {authorized_count}:{unauthorized_count}")
            print("\nSamples per Speaker:")
            for speaker in sorted(speaker_sample_counts.keys()):
                print(f"  {speaker}: {speaker_sample_counts[speaker]}")
            print()
