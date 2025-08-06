"""
This script runs the transcription and diarization pipelines

Steps:
1. Loads audio dataset
2. Preprocesses audio file to reduce present noise
3. Diarizes speakers
4. Transcribes audio
5. Saves the output to disk
"""

import os
import librosa
import soundfile as sf
import tempfile
from datasets import Audio, load_dataset
from torch.cuda import init
from processing import process_audio
from utils import clear_memory
from preprocess.preprocess_pipeline import preprocess_file

# load voxconverse dataset
dataset = load_dataset("diarizers-community/voxconverse", trust_remote_code=True)
test_files = dataset["test"].select(range(1))  # change range to whatever num of audio files you wish to process

# create temp directory
os.makedirs("tmp", exist_ok=True)

# process each sample in the test dataset
for sample in test_files:
    audio_data = sample["audio"]
    waveform = audio_data["array"]
    sr = audio_data["sampling_rate"]

    # generate file path for preprocessed audio files
    audio_id = os.path.splitext(os.path.basename(audio_data["path"]))[0]
    initial_path = os.path.join("tmp", f"{audio_id}_original.wav")
    preprocessed_path = os.path.join("tmp", f"{audio_id}_preprocessed.wav")

    # write original file to temp dir
    sf.write(initial_path, waveform, sr)

    # preprocess audio file and remove the original from temp dir
    preprocess_file(initial_path, preprocessed_path)

    if os.path.exists(initial_path):
        os.remove(initial_path)

    # update sample with preprocessed audio
    waveform, sr = librosa.load(preprocessed_path, sr=None)
    sample["audio"]["path"] = preprocessed_path
    sample["audio"]["array"] = waveform
    sample["audio"]["sampling_rate"] = sr

    # diarize and transcribe audio
    process_audio(sample)
    clear_memory()
    print("Cleared GPU memory between audio files")

print("✅ All files processed!")
