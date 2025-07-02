# Instructions

To run the code, one must first install the required packages for python. A token.txt file containing your HF token
is also required in the parent directory of your cloned project.

1. Core audio processing <br>

```shell
pip install numpy librosa soundfile noisereduce
```

2. Deep learning frameworks

```shell
pip install torch torchaudio
```

3. Hugging Face ecosystem

```shell
pip install transformers datasets
echo $HF_TOKEN >> token.txt
```

*NOTE: Make sure that your hugging face token has access
to [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
and [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)!*

4. Speaker diarization

```shell
pip install pyannote.audio
```

5. Speech processing

```shell
pip install speechbrain
```

6. Utilities

```shell
pip install tqdm
```
