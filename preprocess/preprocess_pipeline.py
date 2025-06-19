"""
This modules is responsible for running the preprocessing pipeline.
It applies normalization, noise reduction, trimming and resampling.
"""

import librosa
import soundfile as sf
import os
from .normalizing import normalize_audio
from .denoising import reduce_noise
from .trim_silence import trim_silence
from .resample import resample_audio


def preprocess_file(input_path, output_path, target_sr=16000):
    """
    This method runs the audio preprocessing.

    Keyword parameters:
    input_path -- path to input audio file
    output_path -- path to save preprocessed audio file
    target_sr -- target sample rate. set to 16KHz by default
    """
    audio, sr = librosa.load(input_path, sr=None)

    audio = normalize_audio(audio)
    audio = reduce_noise(audio, sr)
    audio = trim_silence(audio, sr)
    if sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio, target_sr)
    print(f"✅ Preprocessed: {output_path}")
    ...
