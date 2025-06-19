"""
Audio transcription module.
Supports transcription with both Whisper and Speechbrain based on whatever
was selected in the configuration module.
"""
from random import sample
from fsspec.utils import tempfile
import numpy as np
import torch
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2FeatureEncoder
from config import USE_WHISPER, MAX_AUDIO_LEN_WHISPER
import soundfile as sf
import os

# initialize the transcription model
if USE_WHISPER:
    from transformers import pipeline as hf_pipeline
    transcriber = hf_pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small.en",
        device=0 if torch.cuda.is_available() else -1,
        chunk_length_s=30
    )
else:
    from speechbrain.pretrained import EncoderDecoderASR
    transcriber = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-crdnn-commonvoice-en",
        savedir="pretrained_models/asr-crdnn-commonvoice-en",
        run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )


def transcribe_segment(audio_segment, sample_rate):
    """
    Transcribes an audio segment using the selected model

    Keyword arguments:
    audio_segment -- audio sample as numpy array
    sample_rate -- the audio segment's sample rate
    """
    audio_len = len(audio_segment) / sample_rate
    try:
        if USE_WHISPER:
            segment_dict = {
                "raw": np.array(audio_segment),
                "sampling_rate": sample_rate,
                "language": "en"
            }
            result = transcriber(segment_dict, return_timestamps=(
                audio_len > MAX_AUDIO_LEN_WHISPER))
            return result.get("text", "").strip() if isinstance(result, dict) else str(result).strip()
        else:
            # SpeechBrain requires a file input
            temp_filename = "temp_segment.wav"
            sf.write(temp_filename, audio_segment, sample_rate)
            result = transcriber.transcribe_file(temp_filename).strip()
            os.remove(temp_filename)
            return result
    except Exception as e:
        print(f"Transcription error: {e}")
        return "[ERROR]"


def transcribe_with_timestamps(waveform, sample_rate):
    """
    Transcribes with word-level timestamps. Used with Whisper only.   

    Keyword arguments:
    waveform -- the audio's waveform as numpy array
    sample_rate -- the audio's sample rate
    """
    segment_dict = {
        "raw": np.array(waveform),
        "sampling_rate": sample_rate,
        # "language": "en"
    }

    # gets the segmented transcription
    result = transcriber(
        segment_dict,
        return_timestamps=True,
        # chunk_len_s=30
    )

    return [
        {
            "start": chunk["timestamp"][0],
            "end": chunk["timestamp"][1],
            "text": chunk["text"].strip()
        }
        for chunk in result["chunks"]
    ]
