"""
Core audio processing module.
Manages the diarization and transcription logic.
Supports multiple diarization methods: pyannote and whisper-embedding
"""
import os
import numpy as np
from tqdm import tqdm
import torch
from config import (
    DIARIZATION_METHOD,
    MIN_EMBEDDING_DURATION,
    MIN_SEGMENT_DURATION,
    SPEAKER_SIMILARITY_THRESHOLD,
    USE_DIARIZATION,
    USE_CENTRAL_EMBEDDINGS
)
from diarization import init_diarization_pipeline
from speaker_embedding import SpeakerEmbeddingModel
from transcription import transcribe_segment, transcribe_with_timestamps
from utils import clear_memory, cosine_similarity

# Import clustering library
try:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not found. Clustering for whisper_embedding will not work.")
    SKLEARN_AVAILABLE = False

if USE_CENTRAL_EMBEDDINGS:
    from database import init_central_json
    init_central_json()

# initialize the diarization pipeline if needed
diarization_pipeline = init_diarization_pipeline() if USE_DIARIZATION else None


def process_audio(sample):
    """
    Method for processing the audio file

    Keyword argument:
    sample -- dictionary that contains the audio data and metadata
    """
    print("DEBUG: Entering process_audio function")
    audio_id = os.path.splitext(os.path.basename(sample["audio"]["path"]))[0]
    waveform = sample["audio"]["array"]
    sample_rate = sample["audio"]["sampling_rate"]
    print(f"DEBUG: Processing audio: {audio_id} with length: {len(waveform)/sample_rate:.2f}s")

    # transcribes the audio file using the selected model if no diarization is needed
    if not USE_DIARIZATION:
        print(f"DEBUG: Diarization disabled. Transcribing audio from {audio_id}")
        print(f"Transcribing audio for {audio_id}...")
        try:
            transcription = transcribe_segment(waveform, sample_rate)
            os.makedirs("output", exist_ok=True)
            with open(os.path.join("output", f"{audio_id}_full_transcript.txt"), "w") as f:
                f.write(transcription)
            print(f"DEBUG: Transcription done and saved for {audio_id}")
            return
        except Exception as e:
            print(f"ERROR: Failed during transcription for {audio_id}: {e}")

    print(f"Processing {audio_id} with {DIARIZATION_METHOD} diarization...")

    # runs the pyannote diarization pipeline WITH EMBEDDINGS
    if DIARIZATION_METHOD == "pyannote":
        print(f"DEBUG: Starting pyannote processing for {audio_id}")
        diarization = None
        embeddings = None
        
        # attempts to run the pipeline with the selected batch size
        # reduces batch size in case of "out of memory" error
        for attempt in range(3):
            try:
                print(f"DEBUG: Running pyannote pipeline with return_embeddings=True (attempt {attempt + 1})")
                # THIS IS THE KEY CHANGE - using return_embeddings=True
                result = diarization_pipeline({
                    "waveform": torch.tensor(waveform).float().unsqueeze(0),
                    "sample_rate": sample_rate
                }, return_embeddings=True)
                
                if isinstance(result, tuple) and len(result) == 2:
                    diarization, embeddings = result
                else:
                    diarization = result
                    embeddings = None
                    print("WARNING: return_embeddings=True didn't return embeddings tuple")
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(
                        f"⚠️ OOM (attempt {attempt + 1}) - Reducing batch size")
                    diarization_pipeline.segmentation_batch_size = max(
                        1, diarization_pipeline.segmentation_batch_size // 2)
                    diarization_pipeline.embedding_batch_size = max(
                        1, diarization_pipeline.embedding_batch_size // 2)
                    clear_memory()
                else:
                    raise
            except Exception as e:
                print(f"⚠️ Error with return_embeddings (attempt {attempt + 1}): {e}")
                if attempt == 2:  # Last attempt, try without embeddings
                    print("⚠️ Falling back to pipeline without return_embeddings")
                    try:
                        diarization = diarization_pipeline({
                            "waveform": torch.tensor(waveform).float().unsqueeze(0),
                            "sample_rate": sample_rate
                        })
                        embeddings = None
                        break
                    except Exception as e2:
                        print(f"⚠️ Error without return_embeddings: {e2}")
                        raise
        
        # use CPU if GPU usage is not possible
        if diarization is None:
            print("⚠️ Switching diarization to CPU")
            diarization_pipeline.to("cpu")
            try:
                result = diarization_pipeline({
                    "waveform": torch.tensor(waveform).float().unsqueeze(0),
                    "sample_rate": sample_rate
                }, return_embeddings=True)
                
                if isinstance(result, tuple) and len(result) == 2:
                    diarization, embeddings = result
                else:
                    diarization = result
                    embeddings = None
            except Exception as e:
                print(f"⚠️ Error with return_embeddings on CPU: {e}")
                diarization = diarization_pipeline({
                    "waveform": torch.tensor(waveform).float().unsqueeze(0),
                    "sample_rate": sample_rate
                })
                embeddings = None
            diarization_pipeline.to(torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"))
        clear_memory()

        # Get unique speakers and their embeddings
        unique_speakers = list(diarization.labels()) if hasattr(diarization, 'labels') else []
        print(f"DEBUG: Found {len(unique_speakers)} unique speakers: {unique_speakers}")
        
        if embeddings is not None:
            print(f"DEBUG: Got {len(embeddings)} speaker embeddings from pyannote")
        else:
            print("DEBUG: No embeddings returned from pyannote pipeline")

        # Initialize embedding model for central database operations
        embedding_model = None
        if USE_CENTRAL_EMBEDDINGS:
            embedding_model = SpeakerEmbeddingModel("pyannote")

        # Map speakers to central database IDs
        speaker_id_mapping = {}
        
        if embeddings is not None and USE_CENTRAL_EMBEDDINGS and embedding_model is not None:
            print("DEBUG: Mapping speakers using pyannote embeddings")
            for i, speaker_label in enumerate(unique_speakers):
                if i < len(embeddings):
                    speaker_embedding = embeddings[i]
                    print(f"DEBUG: Processing embedding for speaker {speaker_label}")
                    
                    # Find matching speaker in central database
                    match_id = embedding_model.find_matching_speaker_in_central_db(
                        speaker_embedding, SPEAKER_SIMILARITY_THRESHOLD)
                    
                    if match_id is not None:
                        speaker_id_mapping[speaker_label] = f"SPK_C_{match_id}"
                        print(f"DEBUG: Mapped speaker {speaker_label} to existing ID {match_id}")
                    else:
                        # Save new speaker embedding
                        new_id = embedding_model.save_embedding_to_central_db(speaker_embedding)
                        if new_id is not None:
                            speaker_id_mapping[speaker_label] = f"SPK_C_{new_id}"
                            print(f"DEBUG: Created new speaker {speaker_label} with ID {new_id}")
                        else:
                            speaker_id_mapping[speaker_label] = speaker_label
                            print(f"DEBUG: Failed to save speaker {speaker_label}, using original label")
                else:
                    speaker_id_mapping[speaker_label] = speaker_label
                    print(f"DEBUG: No embedding for speaker {speaker_label}, using original label")
        else:
            # Fallback: just use original speaker labels
            for speaker_label in unique_speakers:
                speaker_id_mapping[speaker_label] = speaker_label
            print("DEBUG: Using original speaker labels (no embedding matching)")

        # process diarization segments with mapped speaker IDs
        segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if segment.duration < MIN_SEGMENT_DURATION:
                continue
            mapped_speaker = speaker_id_mapping.get(speaker, speaker)
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": mapped_speaker,
                "audio": waveform[int(segment.start * sample_rate):int(segment.end * sample_rate)]
            })

        # transcribe each segment
        transcript_lines = []
        for segment in tqdm(segments, desc=f"Transcribing {audio_id}"):
            text = transcribe_segment(segment["audio"], sample_rate)
            transcript_lines.append(
                f"[{segment['start']:.1f}-{segment['end']:.1f}] {segment['speaker']}: {text}")

        # save the transcripted files
        os.makedirs("output", exist_ok=True)
        with open(os.path.join("output", f"{audio_id}_speaker_transcript.txt"), "w") as f:
            f.write("\n".join(transcript_lines))
            
        print(f"✅ Finished processing {audio_id} with pyannote diarization and embedding support")

    # runs the whisper-based diarization using CLUSTERING
    elif DIARIZATION_METHOD == "whisper_embedding":
        if not SKLEARN_AVAILABLE:
            print("ERROR: scikit-learn is required for the 'whisper_embedding' method with clustering.")
            return
            
        print(f"DEBUG: Starting Whisper-based diarization (with clustering) for {audio_id}")
        clear_memory()
        print("DEBUG: Getting timed segments from Whisper")
        try:
            # get timed segments from Whisper
            segments = transcribe_with_timestamps(waveform, sample_rate)
            print(f"DEBUG: Got {len(segments)} segments from Whisper")
        except Exception as e:
            print(f"ERROR: Failed to get Whisper timestamps: {e}")
            return

        print("DEBUG: Initializing SpeakerEmbeddingModel")
        try:
            # initialize embedding model
            embedding_model = SpeakerEmbeddingModel("speechbrain")
            print(f"DEBUG: SpeakerEmbeddingModel initialized for {audio_id}")
        except Exception as e:
            print(f"ERROR: Failed to initialize SpeakerEmbeddingModel for {audio_id}: {e}")
            return

        indexed_segments = [{'original_index': i, **s} for i, s in enumerate(segments)]
        # process longest segments first for better memory usage (good idea!)
        indexed_segments.sort(key=lambda s: s["end"] - s["start"], reverse=True)

        # --- DeepSeek's Solution: Batch Processing with Clustering ---
        
        # First, extract all embeddings
        print("DEBUG: Extracting embeddings for all segments...")
        all_embeddings = []
        valid_segment_indices = [] # Indices in indexed_segments

        for i, seg in enumerate(tqdm(indexed_segments, desc="Extracting embeddings")):
            duration = seg["end"] - seg["start"]
            if duration < MIN_EMBEDDING_DURATION:
                # Mark short segments for later assignment
                seg['speaker'] = "SPK_UNK"
                continue
                
            seg_start = int(seg["start"] * sample_rate)
            seg_end = int(seg["end"] * sample_rate)
            audio_segment = waveform[seg_start:seg_end]
            
            try:
                embedding = embedding_model.get_embedding(audio_segment, sample_rate)
                clear_memory()
                if embedding is not None:
                    all_embeddings.append(embedding)
                    valid_segment_indices.append(i)
                else:
                    # Mark segments with failed embedding extraction
                    seg['speaker'] = "SPK_UNK"
            except Exception as e:
                print(f"Embedding error for segment {i}: {e}")
                seg['speaker'] = "SPK_UNK"

        # Now cluster all embeddings
        if all_embeddings and len(all_embeddings) > 1:
            print(f"DEBUG: Clustering {len(all_embeddings)} embeddings...")
            try:
                # Convert list of arrays to a 2D numpy array for sklearn
                all_embeddings_array = np.array(all_embeddings)
                
                # Use sklearn's cosine_similarity to compute the similarity matrix
                similarity_matrix = sklearn_cosine_similarity(all_embeddings_array)
                # Convert similarity to distance for AgglomerativeClustering
                distance_matrix = 1 - similarity_matrix
                
                # Perform clustering
                # Note: distance_threshold=1-SPEAKER_SIMILARITY_THRESHOLD might be too strict.
                # A common approach is to set a fixed small distance_threshold or use a small n_clusters.
                # Let's try a small fixed distance threshold first, then adjust if needed.
                distance_threshold_for_clustering = 0.6 # e.g., 0.3 if threshold is 0.7
                # Ensure distance_threshold is positive and reasonable
                distance_threshold_for_clustering = max(0.1, min(0.9, distance_threshold_for_clustering))
                
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=distance_threshold_for_clustering, # Use the adjusted threshold
                    metric='precomputed',
                    linkage='average' # 'ward' requires 'euclidean' metric, so use 'average' or 'complete' for 'precomputed'
                )
                
                cluster_labels = clustering.fit_predict(distance_matrix)
                print(f"DEBUG: Clustering resulted in {len(set(cluster_labels))} clusters.")
                
                # Map cluster labels to speaker IDs
                cluster_to_speaker = {}
                next_speaker_id = 0
                
                # Assign speakers to clusters
                for i, cluster_id in enumerate(cluster_labels):
                    original_segment_index = valid_segment_indices[i]
                    
                    if cluster_id not in cluster_to_speaker:
                        # This is the first time we see this cluster_id
                        # Try to find this cluster (represented by its first embedding) in the central database
                        central_match = None
                        if USE_CENTRAL_EMBEDDINGS:
                            central_match = embedding_model.find_matching_speaker_in_central_db(
                                all_embeddings[i], SPEAKER_SIMILARITY_THRESHOLD)
                        
                        if central_match is not None:
                            cluster_to_speaker[cluster_id] = f"SPK_C_{central_match}"
                            print(f"DEBUG: Cluster {cluster_id} matched to existing speaker SPK_C_{central_match}")
                        else:
                            cluster_to_speaker[cluster_id] = f"SPK_{next_speaker_id}"
                            print(f"DEBUG: Created new speaker SPK_{next_speaker_id} for cluster {cluster_id}")
                            next_speaker_id += 1
                            
                            # Save the representative embedding of this NEW cluster to the central database
                            if USE_CENTRAL_EMBEDDINGS:
                                # Use the first embedding of the cluster as the representative
                                embedding_model.save_embedding_to_central_db(all_embeddings[i])
                    
                    # Assign speaker label to the corresponding segment
                    indexed_segments[original_segment_index]['speaker'] = cluster_to_speaker[cluster_id]
                    
            except ValueError as e:
                if "n_samples" in str(e) and "distance_threshold" in str(e):
                    print(f"DEBUG: Clustering failed due to insufficient samples or distance threshold issue: {e}")
                    print("DEBUG: Assigning default speakers...")
                    # Fallback: Assign default speakers if clustering fails
                    speaker_id = 0
                    for i in range(len(all_embeddings)):
                        idx = valid_segment_indices[i]
                        indexed_segments[idx]['speaker'] = f"SPK_{speaker_id}"
                        speaker_id += 1
                        if USE_CENTRAL_EMBEDDINGS:
                            embedding_model.save_embedding_to_central_db(all_embeddings[i])
                else:
                    raise
            except Exception as e:
                print(f"ERROR: Clustering failed with unexpected error: {e}")
                # Fallback: Assign default speakers if clustering fails unexpectedly
                speaker_id = 0
                for i in range(len(all_embeddings)):
                    idx = valid_segment_indices[i]
                    indexed_segments[idx]['speaker'] = f"SPK_{speaker_id}"
                    speaker_id += 1
                    if USE_CENTRAL_EMBEDDINGS:
                        embedding_model.save_embedding_to_central_db(all_embeddings[i])
                        
        elif all_embeddings and len(all_embeddings) == 1:
            print("DEBUG: Only one valid embedding found, assigning it a speaker ID.")
            # If there's only one embedding, it forms its own cluster/speaker
            idx = valid_segment_indices[0]
            if USE_CENTRAL_EMBEDDINGS:
                central_match = embedding_model.find_matching_speaker_in_central_db(
                    all_embeddings[0], SPEAKER_SIMILARITY_THRESHOLD)
                if central_match is not None:
                    indexed_segments[idx]['speaker'] = f"SPK_C_{central_match}"
                else:
                    new_id = embedding_model.save_embedding_to_central_db(all_embeddings[0])
                    indexed_segments[idx]['speaker'] = f"SPK_0" # Assign SPK_0 if new
            else:
                indexed_segments[idx]['speaker'] = "SPK_0"
        else:
            print("DEBUG: No valid embeddings found for clustering.")
            # All segments were marked as SPK_UNK during extraction or none passed the duration check

        # Ensure all segments have a speaker label (should already be handled, but just in case)
        for seg in indexed_segments:
             if 'speaker' not in seg:
                 seg['speaker'] = "SPK_UNK"

        # --- End of Clustering Approach ---

        # Sort segments back to their original order for transcription
        indexed_segments.sort(key=lambda s: s['original_index'])

        # Transcribe each segment
        print("DEBUG: Transcribing segments...")
        transcript_lines = []
        for segment in tqdm(indexed_segments, desc=f"Transcribing {audio_id}"):
            # Extract audio for transcription
            seg_start = int(segment["start"] * sample_rate)
            seg_end = int(segment["end"] * sample_rate)
            audio_segment = waveform[seg_start:seg_end]
            
            try:
                text = transcribe_segment(audio_segment, sample_rate)
                transcript_lines.append(
                    f"[{segment['start']:.1f}-{segment['end']:.1f}] {segment['speaker']}: {text}"
                )
            except Exception as e:
                print(f"ERROR: Failed to transcribe segment [{segment['start']:.1f}-{segment['end']:.1f}]: {e}")
                transcript_lines.append(
                    f"[{segment['start']:.1f}-{segment['end']:.1f}] {segment['speaker']}: [TRANSCRIPTION FAILED]"
                )

        # save the transcript
        os.makedirs("output", exist_ok=True)
        output_filename = f"{audio_id}_speaker_transcript_emb_central_db.txt" if USE_CENTRAL_EMBEDDINGS else f"{audio_id}_speaker_transcript_emb.txt"
        with open(os.path.join("output", output_filename), "w") as f:
            f.write("\n".join(transcript_lines))

        print(f"✅ Finished processing {audio_id} with central JSON integration: {USE_CENTRAL_EMBEDDINGS}")