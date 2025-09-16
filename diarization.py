"""
This module initializes the speaker diarization pipeline
"""

from pyannote.audio import Pipeline
import torch
from config import HF_TOKEN, DEVICE, INITIAL_EMBEDDING_BATCH_SIZE, DIARIZATION_CLUSTERING_THRESHOLD


def init_diarization_pipeline():
    """
    Initialize and configure speaker diarization pipeline

    Returns:
        pipeling: Configured diarization pipeline
    """
    print("DEBUG: Starting pyannote diarization model.")
    try:
        print("DEBUG: Loading pyannote model.")
        # load pretrained diarization model
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)
        print("DEBUG: Loaded pyannote. Configuring the pipeline.")

        # configure the pipeline
        pipeline.to(DEVICE)
        pipeline.segmentation_batch_size = INITIAL_EMBEDDING_BATCH_SIZE
        pipeline.embedding_batch_size = INITIAL_EMBEDDING_BATCH_SIZE
        pipeline.clustering.threshold = DIARIZATION_CLUSTERING_THRESHOLD

        return pipeline
    
    except Exception as e:
        print(f"ERROR: Failed to initialize pyannote pipeline!: {e}")
