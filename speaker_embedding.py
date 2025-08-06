"""
This module extracts the speaker embeddings by using Speechbrain's ECAPA-TDNN
model to generate voice fingerprints.
The module also provides speaker similarity comparison capabilities.
"""
import librosa
import torch
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from config import DEVICE, USE_CENTRAL_EMBEDDINGS
# Import central JSON functions if enabled
if USE_CENTRAL_EMBEDDINGS:
    from database import (
        save_embedding_to_json,
        find_matching_speaker_in_central_json
    )


class SpeakerEmbeddingModel:
    """
    This class represents Speechbrain's ECAPA-TDNN speaker embedding model.
    """

    def __init__(self):
        """
        Initialize speaker embedding model.
        """
        print("DEBUG: Initializing SpeechBrain ECAPA-TDNN speaker embedding model")
        try:
            print("DEBUG: Loading SpeechBrain model")
            self.model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="pretrained_models/spkrec-ecapa-voxceleb",
                run_opts={"device": str(DEVICE)}
            )
            print("DEBUG: SpeechBrain model loaded")
            # freezes the model to improve memory usage (might negatively impact its accuracy)
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            print("DEBUG: SpeechBrain ECAPA-TDNN speaker embedding model initialized")
        except Exception as e:
            print(f"ERROR: Failed to initialize SpeakerEmbeddingModel: {e}")

    def get_embedding(self, audio_segment, sample_rate):
        """
        This method extracts the speaker embedding from an audio segment.

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
                    audio_segment, orig_sr=sample_rate, target_sr=16000)
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
                if len(chunk) < max_chunk_size and len(chunk) > max_chunk_size * 0.1:
                    chunk = np.pad(chunk, (0, max_chunk_size -
                                   len(chunk)), mode='constant')
                elif len(chunk) == 0:
                    continue
                chunks.append(chunk)
            if not chunks:
                if len(audio_segment) > 0:
                    chunks = [audio_segment]
                else:
                    print("⚠️ No valid chunks to embed after processing")
                    return None

            # process the chunks
            embeddings = []
            for chunk in chunks:
                if len(chunk) == 0:
                    continue
                audio_tensor = torch.tensor(
                    chunk).float().unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    # encode_batch typically returns [batch, 1, embedding_dim]
                    chunk_emb = self.model.encode_batch(audio_tensor)
                    chunk_emb_squeezed = chunk_emb.squeeze(1)
                    embeddings.append(chunk_emb_squeezed)
            if not embeddings:
                print("⚠️ No valid chunks to embed")
                return None
            # compute an average value of the embeddings and return it
            embeddings_tensor = torch.cat(embeddings, dim=0)
            avg_embedding = torch.mean(embeddings_tensor, dim=0)
            return avg_embedding.view(-1).cpu().numpy()
        except Exception as e:
            print(f"⚠️ Embedding extraction failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_embedding_to_central_db(self, embedding_np_array):
        if USE_CENTRAL_EMBEDDINGS:
            return save_embedding_to_json(embedding_np_array)
        return None

    def find_matching_speaker_in_central_db(self, query_embedding_np, similarity_threshold):
        if USE_CENTRAL_EMBEDDINGS:
            return find_matching_speaker_in_central_json(query_embedding_np, similarity_threshold)
        return None
