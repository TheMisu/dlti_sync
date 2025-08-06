from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class AudioFile(models.Model):
    file = models.FileField(upload_to='audio_files/')
    original_filename = models.CharField(max_length=255)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.original_filename


class Transcription(models.Model):
    audio_file = models.OneToOneField(AudioFile, on_delete=models.CASCADE)
    content = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # processing status stuff
    PENDING = 'pending'
    PROCESSINGS = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'
    STATUS_CHOICES = [
        (PENDING, 'Pending'),
        (PROCESSINGS, 'Processing'),
        (COMPLETED, 'Completed'),
        (FAILED, 'Failed'),
    ]
    status = models.CharField(
        max_length=32,
        choices=STATUS_CHOICES,
        default=PENDING
    )

    def __str__(self):
        return f"Transcription {self.id} ({self.status})"
