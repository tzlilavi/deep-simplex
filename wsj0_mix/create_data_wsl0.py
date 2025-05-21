
import numpy as np
import tqdm
from numpy.linalg import norm
from numpy.random import uniform
import os
from typing import List, Tuple, Dict
from pathlib import Path

def prepare_speaker_dataset(
    original_data: str = 'csr_1_wav',  # Path to csr_1_wav
    CDs: List[str] = ['11-1.1', '11-2.1'],  # List of CDs ['11-1.1', '11-2.1']
    sentences_origin: str = 'si_tr_s',  # Path to si_tr_s
    train_split: float = 0.8,  # Fraction of speakers in training set
    max_lines_per_speaker: int = 142  # Max unique lines per speaker
) -> Dict[str, Dict[str, List[str]]]:
    """
    Prepare training and validation datasets by splitting speakers and selecting unique lines.

    Args:
        original_data: Path to the "csr_1_wav" folder.
        CDs: List of CD directories containing speakers.
        sentences_origin: Path to the "si_tr_s" folder.
        train_split: Fraction of speakers to allocate to the training set.
        max_lines_per_speaker: Max unique lines to use per speaker.

    Returns:
        A dictionary containing training and validation datasets:
        {
            "train": {speaker_id: [file_paths]},
            "validation": {speaker_id: [file_paths]}
        }
    """
    speaker_files = {}

    # Step 1: Gather all speaker files
    for cd in CDs:
        cd_path = os.path.join(original_data, cd, sentences_origin)
        if not os.path.exists(cd_path):
            raise FileNotFoundError(f"Directory not found: {cd_path}")

        for speaker in os.listdir(cd_path):
            speaker_path = os.path.join(cd_path, speaker)
            if not os.path.isdir(speaker_path):
                continue

            files = [
                os.path.join(speaker_path, file)
                for file in os.listdir(speaker_path)
                if file.endswith(".wav")
            ]

            speaker_id = speaker
            if speaker_id not in speaker_files:
                speaker_files[speaker_id] = []
            speaker_files[speaker_id].extend(files)

    # Step 2: Split speakers into training and validation
    all_speakers = list(speaker_files.keys())
    random.shuffle(all_speakers)

    num_train = int(len(all_speakers) * train_split)
    train_speakers = all_speakers[:num_train]
    validation_speakers = all_speakers[num_train:]

    # Step 3: Select unique lines for each speaker
    def select_lines(files: List[str], max_lines: int) -> List[str]:
        """Select up to max_lines unique lines for a speaker."""
        lines = {}
        selected_files = []

        for file in files:
            line_id = file[-9:-4]  # Extract last 4 characters of line (e.g., "a010a")
            if line_id not in lines:
                lines[line_id] = 0

            if lines[line_id] < 2:  # Allow up to 2 uses of the same line
                selected_files.append(file)
                lines[line_id] += 1

            if len(selected_files) >= max_lines:
                break

        return selected_files

    # Prepare datasets
    dataset = {"train": {}, "validation": {}}

    for speaker_id in train_speakers:
        dataset["train"][speaker_id] = select_lines(speaker_files[speaker_id], max_lines_per_speaker)

    for speaker_id in validation_speakers:
        dataset["validation"][speaker_id] = select_lines(speaker_files[speaker_id], max_lines_per_speaker)

    return dataset
