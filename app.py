from __future__ import annotations

import io
import wave
import uuid
import base64
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torchaudio
import av
import numpy as np
from PIL import Image
from flask import Flask, jsonify, render_template, request, send_file


app = Flask(__name__)


@dataclass
class AudioState:
    waveform: torch.Tensor
    sample_rate: int
    filename: str
    image: Optional[torch.Tensor] = None
    image_filename: Optional[str] = None


def _spectrogram(state: AudioState) -> tuple[torch.Tensor, torch.Tensor]:
    window_length = min(1024, state.waveform.shape[1])
    window_length = max(16, 2 ** int(np.floor(np.log2(window_length))))
    hop_length = max(1, window_length // 4)
    window = torch.hann_window(window_length, device=state.waveform.device)
    transformed = torch.stft(
        state.waveform,
        n_fft=window_length,
        hop_length=hop_length,
        win_length=window_length,
        window=window,
        return_complex=True,
    )
    magnitude = transformed.abs().mean(dim=0)
    frequencies = torch.fft.rfftfreq(window_length, d=1 / state.sample_rate)
    return magnitude, frequencies


SESSIONS: Dict[str, AudioState] = {}


def _session_id() -> str:
    session_id = request.headers.get("X-Audio-Session")
    if not session_id or session_id not in SESSIONS:
        raise ValueError("Upload an audio file before editing.")
    return session_id


def _audio_metadata(state: AudioState) -> dict:
    return {
        "channels": int(state.waveform.shape[0]),
        "samples": int(state.waveform.shape[1]),
        "sample_rate": state.sample_rate,
        "duration": state.waveform.shape[1] / state.sample_rate,
        "filename": state.filename,
        "min": float(state.waveform.min().item()),
        "max": float(state.waveform.max().item()),
    }


def _image_data_url(image: torch.Tensor) -> str:
    pixels = image.detach().cpu().clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    output = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _export(state: AudioState, start: Optional[int] = None, end: Optional[int] = None):
    waveform = state.waveform[:, start:end] if start is not None and end is not None else state.waveform
    output = io.BytesIO()
    pcm = (waveform.cpu().clamp(-1, 1).mul(32767).to(torch.int16).t().contiguous().numpy()).tobytes()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(waveform.shape[0])
        writer.setsampwidth(2)
        writer.setframerate(state.sample_rate)
        writer.writeframes(pcm)
    output.seek(0)
    return output


def _decode_audio(file_stream) -> tuple[torch.Tensor, int]:
    with av.open(file_stream) as container:
        audio_stream = container.streams.audio[0]
        channel_count = int(audio_stream.channels or 0)
        if channel_count < 1:
            raise ValueError("The audio stream does not declare a valid channel count.")

        frames = []
        for frame in container.decode(audio_stream):
            decoded = np.asarray(frame.to_ndarray())
            if decoded.ndim == 1:
                decoded = decoded.reshape(1, -1) if channel_count == 1 else decoded.reshape(-1, channel_count).T
            elif decoded.ndim == 2 and decoded.shape[0] == 1 and channel_count > 1:
                if decoded.shape[1] % channel_count:
                    raise ValueError("The audio frame has an invalid interleaved channel layout.")
                decoded = decoded.reshape(-1, channel_count).T
            elif decoded.ndim == 2 and decoded.shape[0] != channel_count:
                if decoded.shape[1] == channel_count:
                    decoded = decoded.T
                else:
                    raise ValueError("The audio frame has an invalid channel layout.")
            frames.append(decoded)

        if not frames:
            raise ValueError("The uploaded file does not contain audio frames.")
        decoded = np.concatenate(frames, axis=1)
        if decoded.shape[0] != channel_count or decoded.shape[1] == 0:
            raise ValueError("The uploaded file does not contain valid audio samples.")
        if np.issubdtype(decoded.dtype, np.integer):
            decoded = decoded.astype(np.float32) / np.iinfo(decoded.dtype).max
        waveform = torch.from_numpy(decoded).float()
        return waveform, int(audio_stream.rate)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/upload")
def upload():
    audio_file = request.files.get("audio")
    if audio_file is None or not audio_file.filename:
        return jsonify({"error": "Choose a WAV or MP3 file to upload."}), 400

    try:
        waveform, sample_rate = _decode_audio(audio_file.stream)
    except Exception as exc:
        return jsonify({"error": f"Could not decode the audio file: {exc}"}), 400

    if waveform.ndim != 2 or waveform.shape[1] == 0:
        return jsonify({"error": "The uploaded file does not contain audio samples."}), 400

    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = AudioState(
        waveform=waveform.float().contiguous(),
        sample_rate=int(sample_rate),
        filename=audio_file.filename,
    )
    response = jsonify({"session_id": session_id, **_audio_metadata(SESSIONS[session_id])})
    response.headers["X-Audio-Session"] = session_id
    return response


@app.get("/api/spectrogram")
def spectrogram():
    try:
        state = SESSIONS[_session_id()]
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    magnitude, frequencies = _spectrogram(state)
    max_frames = 4000
    if magnitude.shape[1] > max_frames:
        step = (magnitude.shape[1] + max_frames - 1) // max_frames
        magnitude = magnitude[:, ::step]
    magnitude = torch.log1p(magnitude)
    magnitude = magnitude / magnitude.amax().clamp_min(1e-8)
    pitch_bins = magnitude.argmax(dim=0)
    pitches = frequencies[pitch_bins]
    return jsonify({
        "values": magnitude.transpose(0, 1).tolist(),
        "pitches": pitches.tolist(),
        "pitch_magnitudes": magnitude.amax(dim=0).tolist(),
        "frequencies": frequencies.tolist(),
        **_audio_metadata(state),
    })


@app.post("/api/image")
def upload_image():
    try:
        state = SESSIONS[_session_id()]
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    image_file = request.files.get("image")
    if image_file is None or not image_file.filename:
        return jsonify({"error": "Choose a PNG, JPEG, or WEBP image."}), 400

    try:
        with Image.open(image_file.stream) as source:
            rgb = source.convert("RGB")
            pixels = np.asarray(rgb, dtype=np.float32) / 255.0
        image = torch.from_numpy(pixels).permute(2, 0, 1).contiguous()
    except Exception as exc:
        return jsonify({"error": f"Could not decode the image: {exc}"}), 400

    state.image = image
    state.image_filename = image_file.filename
    return jsonify({
        "filename": state.image_filename,
        "width": int(image.shape[2]),
        "height": int(image.shape[1]),
        "image": _image_data_url(image),
    })


@app.post("/api/apply")
def apply_changes():
    try:
        state = SESSIONS[_session_id()]
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    payload = request.get_json(silent=True) or {}
    try:
        gain = float(payload.get("gain", 1))
        pitch = int(payload.get("pitch", 0))
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "Provide numeric gain and pitch values."}), 400

    if not 0.0 <= gain <= 2.0:
        return jsonify({"error": "Gain must be between 0 and 2."}), 400
    if not -12 <= pitch <= 12:
        return jsonify({"error": "Pitch must be between -12 and 12 semitones."}), 400

    state.waveform = (state.waveform * gain).clamp(-1.0, 1.0)
    if pitch:
        state.waveform = torchaudio.functional.pitch_shift(
            state.waveform, state.sample_rate, pitch, n_fft=1024, hop_length=256
        ).clamp(-1.0, 1.0)
    return jsonify(_audio_metadata(state))


@app.post("/api/resample")
def resample():
    try:
        state = SESSIONS[_session_id()]
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    payload = request.get_json(silent=True) or {}
    target_rate = payload.get("sample_rate")
    if not isinstance(target_rate, int) or not 8000 <= target_rate <= 192000:
        return jsonify({"error": "Sample rate must be an integer between 8000 and 192000 Hz."}), 400

    if target_rate != state.sample_rate:
        state.waveform = torchaudio.functional.resample(state.waveform, state.sample_rate, target_rate)
        state.sample_rate = target_rate
    return jsonify(_audio_metadata(state))


@app.get("/api/download")
def download():
    try:
        state = SESSIONS[_session_id()]
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    start = request.args.get("start", type=int)
    end = request.args.get("end", type=int)
    if (start is None) != (end is None):
        return jsonify({"error": "Both selection start and end are required."}), 400
    if start is not None and (start < 0 or end <= start or end > state.waveform.shape[1]):
        return jsonify({"error": "The requested selection is invalid."}), 400

    suffix = "_selection" if start is not None else ""
    name = f"{state.filename.rsplit('.', 1)[0]}{suffix}.wav"
    return send_file(_export(state, start, end), mimetype="audio/wav", as_attachment=True, download_name=name)


if __name__ == "__main__":
    app.run(debug=True)
