"""
Core audio processing module.
Manages the diarization and transcription logic.
Supports multiple diarization methods: pyannote and whisper-embedding
"""

import os
from torch.utils.data import sampler
from tqdm import tqdm
import torch
from config import DIARIZATION_METHOD, MIN_EMBEDDING_DURATION, MIN_SEGMENT_DURATION, SPEAKER_SIMILARITY_THRESHOLD, USE_DIARIZATION
from diarization import init_diarization_pipeline
from speaker_embedding import SpeakerEmbeddingModel
from transcription import transcribe_segment, transcribe_with_timestamps
from utils import clear_memory, cosine_similarity


# initialize the diarization pipeline if needed
diarization_pipeline = init_diarization_pipeline() if USE_DIARIZATION else None


def process_audio(sample):
    """
    Method for processing the audio file

    Keyword argument:
    sample -- dictionary that contains the audio data and metadata
    """
    audio_id = os.path.splitext(os.path.basename(sample["audio"]["path"]))[0]
    waveform = sample["audio"]["array"]
    sample_rate = sample["audio"]["sampling_rate"]

    # transcribes the audio file using the selected model if no diarization is needed
    if not USE_DIARIZATION:
        print(f"Transcribing full audio for {audio_id}...")
        transcription = transcribe_segment(waveform, sample_rate)
        os.makedirs("output", exist_ok=True)
        with open(os.path.join("output", f"{audio_id}_full_transcript.txt"), "w") as f:
            f.write(transcription)
        return

    print(f"Processing {audio_id} with {DIARIZATION_METHOD} diarization...")

    # runs the pyannote diarization pipeline
    if DIARIZATION_METHOD == "pyannote":
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
        clear_memory()
        # get timed segments from Whisper
        segments = transcribe_with_timestamps(waveform, sample_rate)
        embedding_model = SpeakerEmbeddingModel()
        known_speakers = []     # stores the speaker embeddings

        # Store segments with an index to reorder later
        indexed_segments = [{'original_index': i, **s} for i, s in enumerate(segments)]

        # process longest segments first for better memory usage
        indexed_segments.sort(key=lambda s: s["end"] - s["start"], reverse=True)

        # This will store speaker labels but in the sorted order
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

            # match with speakers known so far
            matched = False
            for i, known_emb in enumerate(known_speakers):
                if embedding is None or known_emb is None:
                    continue

                similarity = cosine_similarity(embedding, known_emb)
                if similarity > SPEAKER_SIMILARITY_THRESHOLD:
                    speaker_labels_sorted.append(f"SPK_{i}")
                    matched = True
                    break

            # create a new speaker if no current speaker matches the embedding
            if not matched:
                new_id = len(known_speakers)
                known_speakers.append(embedding)
                speaker_labels_sorted.append(f"SPK_{new_id}")

        # Associate speaker labels with their segments
        for i in range(len(indexed_segments)):
            indexed_segments[i]['speaker'] = speaker_labels_sorted[i]
            
        # Sort back to original order
        indexed_segments.sort(key=lambda s: s['original_index'])

        # generate the transcript
        transcript_lines = [
            f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['speaker']}: {seg['text']}"
            for seg in indexed_segments
        ]

        # save the transcript
        os.makedirs("output", exist_ok=True)
        with open(os.path.join("output", f"{audio_id}_speaker_transcript_emb.txt"), "w") as f:
            f.write("\n".join(transcript_lines))
