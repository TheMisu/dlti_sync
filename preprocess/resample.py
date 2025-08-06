"""
Module for resampling audio files
"""
import librosa

def resample_audio(audio, orig_sr, target_sr=16000):
    """
    Resamples audio to the target sample rate.

    Keyword parameters:
    audio -- input audio signal
    orig_sr -- original sample rate
    target_sr -- target sample rate
    """
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
