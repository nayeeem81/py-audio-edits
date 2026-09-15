# PyTorch Audio Tensor Editor

A local Flask application for visualizing and editing WAV or MP3 audio with PyTorch tensors. The editor converts the `[channels, time steps]` tensor into a magnitude spectrogram with `torch.stft`, overlays the detected pitch, and supports pitch shifting before playback or export. PyAV provides bundled FFmpeg-based decoding for supported input formats.

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start the editor:

```powershell
python app.py
```

Open `http://127.0.0.1:5000` in a browser.

## Editing workflow

1. Upload a WAV or MP3 file.
2. Upload a PNG, JPEG, or WEBP image. The server stores it as a `[3, height, width]` PyTorch tensor.
3. Set the pitch shift, gain, and/or sample rate.
4. Click **Keep changes** to apply the edits to the full audio tensor.
5. Use **Play** or **Download** for the current edited audio. During playback, the image canvas follows the audio time path; pitch frequency changes the color direction and pitch magnitude controls the color-change strength.

Audio is kept in memory for the local process and is preserved with all channels. Exported files are WAV files, regardless of the input format.
