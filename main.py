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
test_files = dataset["test"].select(range(2))  # change range to whatever num of audio files you wish to process

# create temp directory
os.makedirs("tmp", exist_ok=True)

# process each sample in the test dataset
for i, sample in enumerate(test_files):
    print(f"=== Starting processing for sample {i+1}/{len(test_files)} ===")
    audio_data = sample["audio"]

    try:
        decoded_audio_tensor = audio_data.get_all_samples().data
        waveform = decoded_audio_tensor.cpu().numpy()

        sr = audio_data.metadata.sample_rate

        if waveform.ndim > 1 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
        elif waveform.ndim > 1:
            print(f"Warning: Multi-channel audio detected with {waveform.shape[0]} channels. Using first channel.")
            waveform = waveform[0]

    except AttributeError as e:
        print(f"Error accessing audio data or metadata: {e}")
        print(f"AudioDecoder object attributes/methods: {dir(audio_data)}")
        raise

    audio_path = None
    original_audio_info = sample.get("audio", {})
    if isinstance(original_audio_info, dict):
        audio_path = original_audio_info.get("path")

    if not audio_path:
        audio_path = sample.get("path")

    if not audio_path:
        audio_id_generic = sample.get("id", f"sample_{sample.get('__index_level_0__', 'unknown')}")
        audio_id = str(audio_id_generic)
        print(f"Warning: Original audio path not found easily. Using ID: {audio_id}")
    else:
        audio_id = os.path.splittext(os.path.basename(audio_path))[0]

    initial_path = os.path.join("tmp", f"{audio_id}_original.wav")
    preprocessed_path = os.path.join("tmp", f"{audio_id}_preprocessed.wav")

    sf.write(initial_path, waveform, sr)

    preprocess_file(initial_path, preprocessed_path)
    if os.path.exists(initial_path):
        os.remove(initial_path)

    waveform_preprocessed, sr_preprocessed = librosa.load(preprocessed_path, sr=None)

    sample["audio"] = {
        "path": preprocessed_path,
        "array": waveform_preprocessed,
        "sampling_rate": sr_preprocessed
    }

    process_audio(sample)
    clear_memory()
    print("Cleared GPU memory between audio files")


print("✅ All files processed!")
