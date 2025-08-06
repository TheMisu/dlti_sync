import os
import json
import paramiko
from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .forms import AudioUploadForm
from .models import AudioFile, Transcription


# config processing server
GPU_SERVER_HOSTNAME = 'gondor.hucompute.org'
GPU_SERVER_USERNAME = 's2348677'
GPU_SERVER_PASSWORD = 'a&T9uycoBH'
REMOTE_PATH_TO_SCRIPT = '/home/stud_homes/s2348677/webapp_transcript/processing.py'
LOCAL_MEDIA_ROOT = settings.MEDIA_ROOT
REMOTE_STORAGE_PATH  = '/home/stud_homes/s2348677/audioFiles'


# Create your views here.
def landing_page(request):
    return render(request, 'transcription_app/landing.html')


def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('upload_audio')
    else:
        form = UserCreationForm()
    return render(request, 'transcription_app/signup.html', {'form': form})


@login_required
def upload_audio(request):
    if request.method == 'POST':
        form = AudioUploadForm(request.POST, request.FILES)
        if form.is_valid():
            audio_file = form.save(commit=False)
            audio_file.uploaded_by = request.user
            audio_file.original_filename = request.FILES['file'].name
            audio_file.save()


            transcription = Transcription.objects.create(
                audio_file=audio_file,
                content="",
                status=Transcription.PENDING
            )

            # file transfer to gondor
            try:
                local_file_path = audio_file.file.path

                remote_filename = os.path.basename(local_file_path)
                remote_file_path = os.path.join(REMOTE_STORAGE_PATH, remote_filename)

                ssh_client = paramiko.SSHClient()
                ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                ssh_client.connect(GPU_SERVER_HOSTNAME, username=GPU_SERVER_USERNAME, password=GPU_SERVER_PASSWORD)

                sftp_client = ssh_client.open_sftp()
                sftp_client.put(local_file_path, remote_file_path)
                sftp_client.close()

                command = f"CUDA_VISIBLE_DEVICES=3 python {REMOTE_PATH_TO_SCRIPT} --file_path '{remote_file_path}' --transcription_id {transcription.id}"

                stdin, stdout, stderr = ssh_client.exec_command(command)
                ssh_client.close()

                transcription.status = Transcription.PROCESSING
                transcription.save()

                return redirect('landing_page')

            except Exception as e:
                print(f"ERROR during SFTP/SSH: {e}")
                transcription.status = Transcription.FAILED
                transcription.error_message = f"Failed to start processing: {str(e)}"
                audio_file.delete()

                return render(request, 'transcription_app/upload.html', {
                    'form': form,
                    'error': f'ERROR: Failed to start processing: {e}'
                })
            
            # if not PROCESSING_AVAILABLE:
            #     return render(request, 'transcription_app/upload.html', {
            #         'form': form,
            #         'error': 'Audio processing function is not available!'
            #     })
            # 
            # # process the file using the already implemented pipeline
            # try:
            #     file_path = audio_file.file.path
            #
            #     transcription_text = process_audio_from_file(file_path)
            #
            #     transcription = Transcription.objects.create(
            #             audio_file=audio_file,
            #             content=transcription_text
            #         )
            #
            #     # redirect the user to the editing page
            #     return redirect('edit_transcription', transcription_id=transcription.id)
            # except Exception as e:
            #     print(f"ERROR: Audio processing failed: {e}")
            #     return render(request, 'transcription_app/upload.html', {
            #         'form': form,
            #         'error': f'ERROR: Audio processing failed: {e}'
            #     })

    else:
        form = AudioUploadForm()

    return render(request, 'transcription_app/upload.html', {
        'form': form
    })


def worker_callback(request): 
    try:
        data = json.loads(request.body)
        transcription_id = data.get('transcription_id')
        status = data.get('status')
        content = data.get('content', '')
        error_message = data.get('error_message', '')

        if not transcription_id or status not in [Transcription.COMPLETED, Transcription.FAILED]:
            return JsonResponse({
                'error': 'Invalid data provided'
            },
                status=400
            )

        transcription = get_object_or_404(Transcription, id=transcription_id)

        transcription.status = status
        if status == Transcription.COMPLETED:
            transcription.content = content
        elif status == Transcription.FAILED:
            transcription.error_message = error_message

        transcription.save()

        return JsonResponse({
            'message': 'Result received successfully'
        })

    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Transcription.DoesNotExist:
        return JsonResponse({'error': 'Transcription not found'}, status=404)
    except Exception as e:
        print(f"Error in worker_callback: {e}")
        return JsonResponse({'error': 'Internal server error'}, status=500)
        return JsonResponse({'error': 'Transcription not found'}, status=404)

@login_required
def edit_transcription(request, transcription_id):
    transcription = get_object_or_404(Transcription, id=transcription_id)

    if request.method == 'POST':
        data = json.loads(request.body)
        transcription.content = data['content']
        transcription.save()
        return JsonResponse({
            'status': 'success'
        })
        
    return render(request, 'transcription_app/edit_transcription.html', {
        'transcription': transcription
    })
