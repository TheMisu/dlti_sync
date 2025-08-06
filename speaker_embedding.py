"""
This module extracts the speaker embeddings by using Speechbrain's ECAPA-TDNN
model to generate voice fingerprints.
The module also provides speaker similarity comparison capabilities.
"""

import librosa
import torch
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from config import DEVICE


class SpeakerEmbeddingModel:
    """
    This class represents Speechbrain's ECAPA-TDNN speaker embedding model
    """

    def __init__(self):
        """
        Initialize speaker embedding model
        """
        # loads the pretrained model
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxcelab",
            run_opts={"device": str(DEVICE)}
        )

        # freezes the model to improve memory usage (might negatively impact its accuracy)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def get_embedding(self, audio_segment, sample_rate):
        """
        This method extracts the speaker embedding from an audio segment

        Keyword arguments:
        audio_segment -- audio sample as numpy array
        sample_rate -- the audio segment's sample rate
        """
        # returns None in case of an empty segment
        if len(audio_segment) == 0:
            print("⚠️ Empty audio segment. Skipping embedding")
            return None

        # resample the audio file if needed
        if sample_rate != 16000:
            try:
                audio_segment = librosa.resample(
                    audio_segment, orig_st=sample_rate, target_sr=16000)
                sample_rate = 16000
            except Exception as e:
                print(f"⚠️ Resampling failed: {e}")
                return None

        try:
            # split long segments into smaller chunks
            max_chunk_size = 16000 * 10
            chunks = []
            for i in range(0, len(audio_segment), max_chunk_size):
                chunk = audio_segment[i:i + max_chunk_size]

                if len(chunk) < max_chunk_size:
                    chunk = np.pad(chunk, (0, max_chunk_size - len(chunk)))
                chunks.append(chunk)

            if not chunks:
                chunks = [audio_segment]

            # process the chunks
            embeddings = []
            for chunk in chunks:
                if len(chunk) == 0:
                    continue

                audio_tensor = torch.tensor(
                    chunk).float().unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    chunk_emb = self.model.encode_batch(audio_tensor)
                    embeddings.append(chunk_emb)

            if not embeddings:
                print("⚠️ No valid chunks to embed")
                return None

            # compute an average value of the embeddings and return it
            embeddings_tensor = torch.stack(embeddings)
            avg_embedding = torch.mean(embeddings_tensor, dim=0)

            return avg_embedding.view(-1).cpu()

        except Exception as e:
            print(f"⚠️ Embedding extraction failed: {e}")
            return None
