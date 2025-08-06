"""
This module trims/removes silence in the beginning/ending of the audio file
"""

import librosa

def trim_silence(audio, sr, top_db=30):
    """
    This method removes leading/trailing silence from the audio file
    using an energy threshold.

    Keyword parameters:
    audio -- input audio signal
    sr -- sampling rate
    top_db -- threshold in DB below reference

    Returns:
    ndarray: the trimmed audio
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
    return trimmed
