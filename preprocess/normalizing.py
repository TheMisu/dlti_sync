"""
Audio normalization module.
Implements normalization based on peak amplitude
"""

import numpy as np


def normalize_audio(audio):
    """
    Normaliza audio to [-1, 1] range based on peak amplitude

    Keyword parameters:
    audio -- input audio signal

    Returns: 
    ndarray: normalized audio
    """
    peak = np.abs(audio).max()
    return audio / peak if peak > 0 else audio
