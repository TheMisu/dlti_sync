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

    # runs the pyannote diarization pipeline
    if DIARIZATION_METHOD == "pyannote":
        print(f"DEBUG: Starting pyannote processing for {audio_id}")
        diarization = None
        # attempts to run the pipeline with the selected batch size
        # reduces batch size in case of "out of memory" error
        for attempt in range(3):
            try:
                diarization = diarization_pipeline({
                    "waveform": torch.tensor(waveform).float().unsqueeze(0),
                    "sample_rate": sample_rate
                })
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
        # use CPU if GPU usage is not possible
        if diarization is None:
            print("⚠️ Switching diarization to CPU")
            diarization_pipeline.to("cpu")
            diarization = diarization_pipeline({
                "waveform": torch.tensor(waveform).float().unsqueeze(0),
                "sample_rate": sample_rate
            })
            diarization_pipeline.to(torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"))
        clear_memory()

        # process diarization segments
        segments = []
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if segment.duration < MIN_SEGMENT_DURATION:
                continue
            segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker,
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

    # runs the whisper-based diarization
    elif DIARIZATION_METHOD == "whisper_embedding":
        print(f"DEBUG: Starting Whisper-based diarization for {audio_id}")
        clear_memory()
        print("DEBUG: Getting timed segments from Whisper")
        try:
            # get timed segments from Whisper
            segments = transcribe_with_timestamps(waveform, sample_rate)
            print(f"DEBUG: Got {len(segments)} segments from Whisper")
        except Exception as e:
            print(f"ERROR: Failed to get Whisper timestamps: {e}")

        print("DEBUG: Initializing SpeakerEmbeddingModel")
        try:
            # initialize embedding model
            embedding_model = SpeakerEmbeddingModel()
            print(f"DEBUG: SpeakerEmbedddingModel initialized for {audio_id}")
        except Exception as e:
            print(f"ERROR: Failed to initialize SpeakerEmbeddingModel for {audio_id}: {e}")
            

        known_speakers_embeddings = []
        known_speakers_labels = []
        next_local_speaker_id = 0

        indexed_segments = [{'original_index': i, **s}
                            for i, s in enumerate(segments)]
        # process longest segments first for better memory usage (good idea!)
        indexed_segments.sort(
            key=lambda s: s["end"] - s["start"], reverse=True)

        speaker_labels_sorted = []

        for seg in tqdm(indexed_segments, desc=f"Diarizing and embedding {audio_id}"):
            duration = seg["end"] - seg["start"]
            seg_start = int(seg["start"] * sample_rate)
            seg_end = int(seg["end"] * sample_rate)
            audio_segment = waveform[seg_start:seg_end]

            # skip very short segments (usually noise)
            if duration < MIN_EMBEDDING_DURATION:
                speaker_labels_sorted.append("SPK_UNK")
                continue

            # extract speaker embeddings
            try:
                embedding = embedding_model.get_embedding(
                    audio_segment, sample_rate)
                clear_memory()
                if embedding is None:
                    speaker_labels_sorted.append("SPK_UNK")
                    continue
            except Exception as e:
                print(f"Embedding error: {e}")
                speaker_labels_sorted.append("SPK_UNK")
                continue

            matched = False
            assigned_label = "SPK_UNK"

            for i, (local_label, local_emb) in enumerate(zip(known_speakers_labels, known_speakers_embeddings)):
                if embedding is None or local_emb is None:
                    continue
                similarity = cosine_similarity(embedding, local_emb)
                if not np.isnan(similarity):
                    if similarity > SPEAKER_SIMILARITY_THRESHOLD:
                        assigned_label = local_label
                        matched = True
                        break
                else:
                    print(f"Invalid similarity (NaN) between segment and {local_label}. Skipping current comparison")

            if not matched and USE_CENTRAL_EMBEDDINGS:
                central_match_id = embedding_model.find_matching_speaker_in_central_db(
                    embedding, SPEAKER_SIMILARITY_THRESHOLD)
                if central_match_id is not None:
                    assigned_label = f"SPK_C_{central_match_id}"
                    known_speakers_labels.append(assigned_label)
                    known_speakers_embeddings.append(embedding)
                    matched = True

            if not matched:
                assigned_label = f"SPK_{next_local_speaker_id}"
                known_speakers_labels.append(assigned_label)
                known_speakers_embeddings.append(embedding)
                next_local_speaker_id += 1

                if USE_CENTRAL_EMBEDDINGS:
                    central_id = embedding_model.save_embedding_to_central_db(
                        embedding)
                    if central_id is not None:
                        pass

            speaker_labels_sorted.append(assigned_label)

        for i in range(len(indexed_segments)):
            indexed_segments[i]['speaker'] = speaker_labels_sorted[i]

        indexed_segments.sort(key=lambda s: s['original_index'])

        # generate the transcript
        transcript_lines = [
            f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['speaker']}: {seg['text']}"
            for seg in indexed_segments
        ]

        # save the transcript
        os.makedirs("output", exist_ok=True)
        output_filename = f"{audio_id}_speaker_transcript_emb_central_db.txt" if USE_CENTRAL_EMBEDDINGS else f"{audio_id}_speaker_transcript_emb.txt"
        with open(os.path.join("output", output_filename), "w") as f:
            f.write("\n".join(transcript_lines))

        print(f"✅ Finished processing {audio_id} with central JSON integration: {USE_CENTRAL_EMBEDDINGS}")
