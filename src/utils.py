import re
import torch
import torchaudio
import json
import os

# Regex for cleaning text (preserving isiZulu characters)
CHARS_TO_REMOVE_REGEX = r'[!"#$%&\'()*+,-./:;<=>?@\[\\\]^_`{|}~0-9]'

def remove_special_characters(batch):
    """
    Normalizes text by removing special characters and converting to lowercase.
    Expects a dictionary/batch with a 'transcription' key.
    """
    if batch["transcription"] is not None:
        batch["transcription"] = re.sub(CHARS_TO_REMOVE_REGEX, '', batch["transcription"]).lower()
    else:
        batch["transcription"] = ""
    return batch

def filter_duration(example, min_seconds=1.0, max_seconds=20.0, audio_column="audio"):
    """
    Filters audio files based on their duration to prevent OOM errors.
    """
    if example[audio_column] is None or example[audio_column]["array"] is None:
        return False
    # Calculate duration: samples / sampling_rate
    duration = len(example[audio_column]["array"]) / example[audio_column]["sampling_rate"]
    return min_seconds <= duration <= max_seconds

def save_vocab(vocab_dict, output_dir):
    """Saves the vocabulary dictionary to a json file."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'vocab.json'), 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)