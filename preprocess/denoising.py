"""
Module that reduces the noise present in an audio file.
"""
import noisereduce as nr
import numpy as np
import librosa


def estimate_noise(audio, sr, frame_len=2048, hop_len=512, energy_threshold=0.01):
    """
    Estimates the noise from low-energy frames

    Keyword parameters:
    audio -- input audio signal
    sr -- sample rate
    frame_len -- frame length used for energy calculation
    hop_len -- hop length between frames
    energy_threshold -- energy threshold for noise frames

    Returns:
    ndarray: concatenated noise samples or None
    """
    energies = librosa.feature.rms(
        y=audio, frame_length=frame_len, hop_length=hop_len)[0]
    low_energy_frames = np.where(energies < energy_threshold)[0]

    if len(low_energy_frames) == 0:
        print("⚠️ No low-energy frames found for noise profiling.")
        print("⚠️ Skipping noise reduction!")
        return None

    # extract noise samples
    noise_samples = []
    for i in low_energy_frames:
        start = i * hop_len
        end = start + frame_len
        if end <= len(audio):
            noise_samples.append(audio[start:end])

    noise_profile = np.concatenate(noise_samples)
    return noise_profile


def reduce_noise(audio, sr):
    """
    Applies noise reduction to audio files using spectral gating.

    Keyword parameters:
    audio -- input audio signal
    sr -- sample rate

    Returns:
    ndarray: noise-reduced audio
    """
    noise_clip = estimate_noise(audio, sr)
    if noise_clip is None:
        return audio

    reduced_audio = nr.reduce_noise(y=audio, sr=sr, y_noise=noise_clip)
    return reduced_audio.astype(np.float32)
