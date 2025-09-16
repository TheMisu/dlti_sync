import argparse
import sys
import os
import requests
import traceback

DJANGO_CALLBACK_URL = "ENTER URL HERE"

from processing import process_audio_from_file


def send_result(transcription_id, status, content="", error_message=""):
    try:
        data = {
            'transcription_id': transcription_id,
            'status': status,
            'content': content,
            'error_message': error_message
        }
        response = requests.post(DJANGO_CALLBACK_URL, json=data)
        response.raise_for_status()
        print(f"Successfully reported result for transcription {transcription_id}")

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to send result for transcription {transcription_id} to {DJANGO_CALLBACK_URL}: {e}")
    except Exception as e:
        print(f"ERROR: Unexpected error sending result: {e}")


def main():
    parser = argparse.ArgumentParser(description="Proces audio file for transcription.")
    parser.add_argument('--file_path', type=str, required=True, help='Path to the audio file on the GPU server.')
    parser.add_argument('--transcription_id', type=int, required=True, help='ID of the transcription job in Django.')

    args = parser.parse_args()

    file_path = args.file_path
    transcription_id = args.transcription_id

    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        print(f"ERROR: {error_msg}")
        send_result(transcription_id, 'failed', error_message=error_msg)
        sys.exit(1)


    try:
        print(f"Starting processing for transcription ID {transcription_id}, file {file_path}")
        transcription_text = process_audio_from_file(file_path)
        print(f"Processing completed for transcription ID {transcription_id}")
        send_result(transcription_id, 'completed', content=transcription_text)

    except Exception as e:
        error_msg = f"Processing failed: {type(e).__name__}: {str(e)}\nTraceback:\n{''.join(traceback.format_tb(e.__traceback__))}"
        print(f"ERROR: {error_msg}")
        send_result(transcription_id, 'failed', error_message=error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
