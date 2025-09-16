"""
This module enables the speaker embeddings storage in JSON files.
Supports a single unified JSON (old format) and separate JSONs per embedding type (new format).
"""
import os
import json
import numpy as np
import time
import fcntl
from config import CENTRAL_EMBEDDINGS_JSON_PATH, USE_CENTRAL_EMBEDDINGS
from utils import cosine_similarity

CENTRAL_JSON_LOCK_TIMEOUT = 30          # wait 30s for file lock
CENTRAL_JSON_LOCK_RETRY_INTERVAL = 0.1  # seconds between lock attempts
PYANNOTE_EMBEDDINGS_JSON_PATH = os.path.join(os.path.dirname(CENTRAL_EMBEDDINGS_JSON_PATH), "pyannote_embeddings.json")
SPEECHBRAIN_EMBEDDINGS_JSON_PATH = os.path.join(os.path.dirname(CENTRAL_EMBEDDINGS_JSON_PATH), "speechbrain_embeddings.json")


def _get_embedding_file_path(embedding_type):
    if embedding_type == "pyannote":
        return PYANNOTE_EMBEDDINGS_JSON_PATH
    elif embedding_type == "speechbrain":
        return SPEECHBRAIN_EMBEDDINGS_JSON_PATH
    else:
        return CENTRAL_EMBEDDINGS_JSON_PATH


def init_embedding_json(embedding_type):
    json_path = _get_embedding_file_path(embedding_type)
    if not os.path.exists(json_path):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump([], f, indent=4)
        print(f"Initialized {embedding_type} embeddings JSON file at {json_path}")


def save_embedding_to_separate_json(embedding_array, embedding_type, speaker_id=None):
    if not USE_CENTRAL_EMBEDDINGS or embedding_array is None:
        return None

    json_path = _get_embedding_file_path(embedding_type)

    try:
        serializable_emb = embedding_to_serializable(embedding_array)
    except ValueError as e:
        print(f"Cannot serialize embedding. Error: {e}")
        return None

    embeddings_list = load_embeddings_safe(json_path)

    if speaker_id is not None:
        for entry in embeddings_list:
            if entry['id'] == speaker_id:
                if 'embeddings' not in entry:
                    entry['embeddings'] = {}
                if 'sources' not in entry:
                    entry['sources'] = []

                entry['embeddings'][embedding_type] = serializable_emb
                if embedding_type not in entry['sources']:
                    entry['sources'].append(embedding_type)

                if save_embeddings_safe(json_path, embeddings_list):
                    print(f"Added {embedding_type} embedding to existing speaker ID {speaker_id} in {json_path}")
                    return speaker_id
                else:
                    print(f"Failed to update speaker ID {speaker_id} in {json_path}")
                    return None
        print(f"Speaker ID {speaker_id} not found in {json_path}")
        return None
    else:
        if embeddings_list:
            new_id = max((emb.get('id', -1) for emb in embeddings_list), default=-1) + 1
        else:
            new_id = 0

        new_entry = {
            "id": new_id,
            "speaker_label": f"SPK_{new_id}",
            "embeddings": {embedding_type: serializable_emb},
            "sources": [embedding_type]
        }
        embeddings_list.append(new_entry)

        if save_embeddings_safe(json_path, embeddings_list):
            print(f"Created new speaker ID {new_id} with {embedding_type} embedding in {json_path}")
            return new_id
        else:
            print(f"Failed to create new speaker with {embedding_type} embedding in {json_path}")
            return None

def load_embeddings_by_type(embedding_type):
    if not USE_CENTRAL_EMBEDDINGS:
        return []
    json_path = _get_embedding_file_path(embedding_type)
    return load_embeddings_safe(json_path)

def find_matching_speaker_in_type_json(query_embedding_np, similarity_threshold, embedding_type, target_speaker_id=None):
    if not USE_CENTRAL_EMBEDDINGS or query_embedding_np is None:
        return None

    json_path = _get_embedding_file_path(embedding_type)
    embeddings_list = load_embeddings_safe(json_path)
    if not embeddings_list:
        print(f"{embedding_type} JSON ({json_path}) is empty or could not be loaded.")
        return None

    if target_speaker_id is not None:
        for entry in embeddings_list:
            if entry['id'] == target_speaker_id and embedding_type in entry.get('embeddings', {}):
                stored_emb_list = entry['embeddings'][embedding_type]
                try:
                    stored_emb_np = _embedding_from_serializable(stored_emb_list)
                    if stored_emb_np is not None:
                        similarity = cosine_similarity(query_embedding_np, stored_emb_np)
                        if similarity >= similarity_threshold:
                            return target_speaker_id
                except Exception as e:
                    print(f"Error processing embedding for speaker ID {target_speaker_id} in {json_path}: {e}")
        return None

    best_match_id = None
    best_similarity = -1.0

    for entry in embeddings_list:
        embeddings_dict = entry.get('embeddings', {})
        if embedding_type in embeddings_dict:
            stored_emb_list = embeddings_dict[embedding_type]
            try:
                stored_emb_np = _embedding_from_serializable(stored_emb_list)
                if stored_emb_np is not None:
                    similarity = cosine_similarity(query_embedding_np, stored_emb_np)
                    if similarity > best_similarity and similarity >= similarity_threshold:
                        best_similarity = similarity
                        best_match_id = entry['id']
            except Exception as e:
                print(f"Error processing embedding ID {entry['id']} from {json_path}: {e}")

    if best_match_id is not None:
        print(f"Found {embedding_type} JSON match (Speaker ID: {best_match_id}, Similarity: {best_similarity:.4f}) in {json_path}")
    else:
        print(f"No matching speaker found in {embedding_type} JSON ({json_path}) above threshold.")

    return best_match_id


def init_central_json():
    if USE_CENTRAL_EMBEDDINGS:
        init_embedding_json("pyannote")
        init_embedding_json("speechbrain")
        # if not os.path.exists(CENTRAL_EMBEDDINGS_JSON_PATH):
        #     os.makedirs(os.path.dirname(CENTRAL_EMBEDDINGS_JSON_PATH), exist_ok=True)
        #     with open(CENTRAL_EMBEDDINGS_JSON_PATH, 'w') as f:
        #         json.dump([], f, indent=4)
        #     print(f"Initialized central embeddings JSON file at {CENTRAL_EMBEDDINGS_JSON_PATH}")


def save_embedding_to_json(embedding_array, embedding_type="speechbrain", speaker_id=None):
    if USE_CENTRAL_EMBEDDINGS:
        return save_embedding_to_separate_json(embedding_array, embedding_type, speaker_id)
    else:
        return None


def load_central_embeddings():
    if not USE_CENTRAL_EMBEDDINGS:
        return []
    
    return load_embeddings_by_type("speechbrain")


def embedding_to_serializable(embedding_np_array):
    if isinstance(embedding_np_array, np.ndarray):
        return embedding_np_array.tolist()
    else:
        raise ValueError("Embedding must be a numpy array")


def _embedding_from_serializable(embedding_list):
    try:
        return np.array(embedding_list, dtype=np.float32)
    except Exception as e:
        print(f"Error converting embedding from JSON: {e}")
        return None


def load_embeddings_safe(json_path):
    if not os.path.exists(json_path):
        return []
    try:
        with open(json_path, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
            data = json.load(f)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return data
    except (IOError, OSError) as e:
        if e.errno in (fcntl.EAGAIN, fcntl.EACCES):
            print(f"Could not acquire read lock on {json_path} (might be locked for writing).")
        else:
            print(f"Error reading {json_path}: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {json_path}: {e}")
        return []


def save_embeddings_safe(json_path, embeddings_list):
    try:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        with open(json_path, 'w') as f:
            start_time = time.time()
            while time.time() - start_time < CENTRAL_JSON_LOCK_TIMEOUT:
                try:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    json.dump(embeddings_list, f, indent=4)
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    return True
                except (IOError, OSError) as e:
                    if e.errno in (fcntl.EAGAIN, fcntl.EACCES):
                        time.sleep(CENTRAL_JSON_LOCK_RETRY_INTERVAL)
                    else:
                        raise
            print(f"Timeout acquiring write lock for {json_path}")
            return False
    except Exception as e:
        print(f"Error saving to {json_path}: {e}")
        return False


def find_matching_speaker_in_central_json(query_embedding_np, similarity_threshold, embedding_type=None, target_speaker_id=None):
    if not USE_CENTRAL_EMBEDDINGS or query_embedding_np is None:
        return None

    if embedding_type:
        return find_matching_speaker_in_type_json(query_embedding_np, similarity_threshold, embedding_type, target_speaker_id)
    else:
        print("No embedding type specified for separate file search.")
        return None