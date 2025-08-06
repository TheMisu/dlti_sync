"""
Configuration module for the transcription and diarization pipeline
Sets global settings and hardware configuration for the used models
"""

import torch
import os

# reduce pytorch GPU memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# functionality switches
USE_WHISPER = True
USE_DIARIZATION = True

# diarization config
DIARIZATION_METHOD = "pyannote"  # Options: "pyannote" or "whisper_embedding"
MIN_EMBEDDING_DURATION = 0.5  # Min audio duration for speaker embedding
SPEAKER_SIMILARITY_THRESHOLD = 0.7  # Cosine similarity threshold for speaker matching
MIN_SEGMENT_DURATION = 0.1  # Min diarization segment duration in seconds
INITIAL_SEGMENTATION_BATCH_SIZE = 2  # batch size for segmentation model
INITIAL_EMBEDDING_BATCH_SIZE = 4  # batch size for embedding model
DIARIZATION_CLUSTERING_THRESHOLD = 0.5  # clustering similarity threshold

# whisper config
MAX_AUDIO_LEN_WHISPER = 30  # duration in seconds

# huggingface token
with open("token.txt") as f:
    HF_TOKEN = f.readline().strip()

# processing device. uses GPU if cuda is available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEBUG: Configured DEVICE in config.py: {DEVICE}")

# config for json storage
CENTRAL_EMBEDDINGS_JSON_PATH = "./database/speaker_embeddings.json"
USE_CENTRAL_EMBEDDINGS = True
