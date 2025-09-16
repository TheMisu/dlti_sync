"""
This module extracts the speaker embeddings by using Speechbrain's ECAPA-TDNN
model to generate voice fingerprints.
The module also provides speaker similarity comparison capabilities.
Supports saving embeddings to separate JSON files based on model type.
"""
import librosa
import torch
import numpy as np
from speechbrain.pretrained import EncoderClassifier
from config import DEVICE, USE_CENTRAL_EMBEDDINGS

# Import central JSON functions if enabled
if USE_CENTRAL_EMBEDDINGS:
    from database import (
        save_embedding_to_separate_json, # Use the new function
        find_matching_speaker_in_type_json, # Use the new function
        init_embedding_json # To ensure file is initialized
    )


class SpeakerEmbeddingModel:
    """
    This class represents speaker embedding models.
    Supports both Speechbrain's ECAPA-TDNN and pyannote embeddings.
    """

    def __init__(self, model_type="speechbrain"):
        """
        Initialize speaker embedding model.

        Args:
            model_type: "speechbrain" or "pyannote"
        """
        self.model_type = model_type
        print(f"DEBUG: Initializing {model_type} speaker embedding model")

        # Ensure the specific JSON file for this type is initialized
        if USE_CENTRAL_EMBEDDINGS:
             init_embedding_json(self.model_type)

        if model_type == "speechbrain":
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
                self.model = None # Explicitly set to None on failure
        elif model_type == "pyannote":
            try:
                # Initialize pyannote embedding model
                from speechbrain.inference.speaker import SpeakerEncoder
                self.model = SpeakerEncoder.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir="pretrained_models/spkrec-ecapa-voxceleb"
                )
                print("DEBUG: Pyannote speaker embedding model initialized")
            except Exception as e:
                print(f"ERROR: Failed to initialize Pyannote SpeakerEmbeddingModel: {e}")
                self.model = None # Explicitly set to None on failure

    def get_embedding(self, audio_segment, sample_rate):
        """
        This method extracts the speaker embedding from an audio segment.

        Keyword arguments:
        audio_segment -- audio sample as numpy array
        sample_rate -- the audio segment's sample rate
        """
        # Check if model was successfully loaded
        if not hasattr(self, 'model') or self.model is None:
            print(f"⚠️ Model not initialized for {self.model_type}. Cannot extract embedding.")
            return None

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

        if self.model_type == "speechbrain":
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
                print(f"⚠️ SpeechBrain embedding extraction failed: {e}")
                import traceback
                traceback.print_exc()
                return None
        elif self.model_type == "pyannote":
            try:
                # Pyannote embedding extraction
                audio_tensor = torch.tensor(audio_segment).float()
                if len(audio_tensor.shape) == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)

                with torch.no_grad():
                    # encode_batch returns [batch, 1, embedding_dim]
                    embedding = self.model.encode_batch(audio_tensor)
                    return embedding.squeeze().cpu().numpy()
            except Exception as e:
                print(f"⚠️ Pyannote embedding extraction failed: {e}")
                import traceback
                traceback.print_exc()
                return None

    def save_embedding_to_central_db(self, embedding_np_array, speaker_id=None):
        """
        Save the extracted embedding to the central database (separate file based on model type).

        Args:
            embedding_np_array: The numpy array of the embedding.
            speaker_id: Optional ID to add this embedding to an existing speaker entry.

        Returns:
            int: The speaker ID assigned by the database, or None on failure.
        """
        if USE_CENTRAL_EMBEDDINGS:
            # Explicitly pass self.model_type to ensure correct file is used
            return save_embedding_to_separate_json(embedding_np_array, self.model_type, speaker_id)
        return None

    def find_matching_speaker_in_central_db(self, query_embedding_np, similarity_threshold):
        """
        Find a matching speaker in the central database (type-specific file).

        Args:
            query_embedding_np: The numpy array of the query embedding.
            similarity_threshold: The minimum similarity score for a match.

        Returns:
            int: The ID of the matching speaker, or None if no match is found.
        """
        if USE_CENTRAL_EMBEDDINGS:
            # Explicitly pass self.model_type to ensure correct file is used
            return find_matching_speaker_in_type_json(query_embedding_np, similarity_threshold, self.model_type)
        return None