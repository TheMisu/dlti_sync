"""
This module enables the speaker embeddings storage in a JSON file.
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


def init_central_json():
    if USE_CENTRAL_EMBEDDINGS and not os.path.exists(CENTRAL_EMBEDDINGS_JSON_PATH):
        os.makedirs(os.path.dirname(
            CENTRAL_EMBEDDINGS_JSON_PATH), exist_ok=True)
        with open(CENTRAL_EMBEDDINGS_JSON_PATH, 'w') as f:
            json.dump([], f, indent=4)
        print(f"Initialized central embeddings JSON file at {CENTRAL_EMBEDDINGS_JSON_PATH}")


def save_embedding_to_json(embedding_array):
    if not USE_CENTRAL_EMBEDDINGS or embedding_array is None:
        return None

    try:
        serializable_emb = embedding_to_serializable(embedding_array)
    except ValueError as e:
        print(f"Cannot serialize embedding. Error: {e}")
        return None

    embeddings_list = load_embeddings_safe(CENTRAL_EMBEDDINGS_JSON_PATH)

    if embeddings_list:
        new_id = max(emb['id'] for emb in embeddings_list) + 1
    else:
        new_id = 0

    new_entry = {"id": new_id, "embedding": serializable_emb}
    embeddings_list.append(new_entry)

    if save_embeddings_safe(CENTRAL_EMBEDDINGS_JSON_PATH, embeddings_list):
        print(f"Saved new embedding ID {new_id} to central JSON.")
        return new_id
    else:
        print("Failed to save embedding to central JSON.")
        return None


def load_central_embeddings():
    if not USE_CENTRAL_EMBEDDINGS:
        return []
    return load_embeddings_safe(CENTRAL_EMBEDDINGS_JSON_PATH)


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
        if e.errno == fcntl.EAGAIN or e.errno == fcntl.EACCES:
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
                    if e.errno == fcntl.EAGAIN or e.errno == fcntl.EACCES:
                        time.sleep(CENTRAL_JSON_LOCK_RETRY_INTERVAL)
                    else:
                        raise
            print(f"Timeout acquiring write lock for {json_path}")
            return False
    except Exception as e:
        print(f"Error saving to {json_path}: {e}")
        return False


def find_matching_speaker_in_central_json(query_embedding_np, similarity_threshold):
    if not USE_CENTRAL_EMBEDDINGS or query_embedding_np is None:
        return None

    embeddings_list = load_central_embeddings()
    if not embeddings_list:
        print("Central JSON is empty or could not be loaded.")
        return None

    best_match_id = None
    best_similarity = -1.0

    for entry in embeddings_list:
        stored_emb_list = entry.get('embedding')
        if stored_emb_list:
            try:
                stored_emb_np = _embedding_from_serializable(stored_emb_list)
                if stored_emb_np is not None:
                    similarity = cosine_similarity(
                        query_embedding_np, stored_emb_np)
                    if similarity > best_similarity and similarity >= similarity_threshold:
                        best_similarity = similarity
                        best_match_id = entry['id']
            except Exception as e:
                print(f"Error processing embedding ID {entry['id']} from JSON: {e}")

    if best_match_id is not None:
        print(f"Found central JSON match (ID: {best_match_id}, Similarity: {best_similarity:.4f})")
    else:
        print("No matching speaker found in central JSON above threshold.")

    return best_match_id
