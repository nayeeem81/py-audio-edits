const fileInput = document.querySelector("#audio-file");
const uploadButton = document.querySelector("#upload-button");
const imageFileInput = document.querySelector("#image-file");
const imageUploadButton = document.querySelector("#image-upload-button");
const imageCanvas = document.querySelector("#image-canvas");
const imageNameEl = document.querySelector("#image-name");
const canvas = document.querySelector("#waveform");
const waveformScroll = document.querySelector("#waveform-scroll");
const gainSlider = document.querySelector("#gain-slider");
const pitchSlider = document.querySelector("#pitch-slider");
const pitchBrightnessSlider = document.querySelector("#pitch-brightness-slider");
const rateSlider = document.querySelector("#rate-slider");
const saveButton = document.querySelector("#save-button");
const breakTensorButton = document.querySelector("#break-tensor-button");
const breakDurationInput = document.querySelector("#break-duration");
const randomTensorButton = document.querySelector("#random-tensor-button");
const playButton = document.querySelector("#play-button");
const playRandomButton = document.querySelector("#play-random-button");
const downloadButton = document.querySelector("#download-selection");
const statusEl = document.querySelector("#status");
const metadataEl = document.querySelector("#metadata");
const errorEl = document.querySelector("#error");
const fileNameEl = document.querySelector("#file-name");
const gainValue = document.querySelector("#gain-value");
const pitchValue = document.querySelector("#pitch-value");
const pitchBrightnessValue = document.querySelector("#pitch-brightness-value");
const rateValue = document.querySelector("#rate-value");

let sessionId = null;
let metadata = null;
let spectrogram = [];
let pitches = [];
let pitchMagnitudes = [];
let randomSegments = [];
let randomStackSpectrogram = [];
let sourceImageData = null;
let audioContext = null;
let playback = null;
let activeRandomSegment = -1;
const PIXELS_PER_FRAME = 3;

function setError(message = "") { errorEl.textContent = message; }
function setStatus(message) { if (statusEl) statusEl.textContent = message; }
function headers() { return { "Content-Type": "application/json", "X-Audio-Session": sessionId }; }

async function requestJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let body;
  try {
    body = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`The server returned an invalid response (${response.status}).`);
  }
  if (!response.ok) throw new Error(body.error || "The request failed.");
  return body;
}

function updateLabels() {
  gainValue.textContent = `${Number(gainSlider.value).toFixed(2)}x`;
  pitchBrightnessValue.textContent = `${pitchBrightnessSlider.value}%`;
  rateValue.textContent = `${rateSlider.value} Hz`;
}

function renderImageCanvas(progress = 0) {
  if (!sourceImageData || !metadata) return;
  const width = sourceImageData.width;
  const height = sourceImageData.height;
  imageCanvas.width = width;
  imageCanvas.height = height;
  const context = imageCanvas.getContext("2d");
  const frame = Math.min(pitches.length - 1, Math.floor(progress * Math.max(0, pitches.length - 1)));
  const magnitude = pitchMagnitudes[frame] || 0;
  const transformed = new ImageData(new Uint8ClampedArray(sourceImageData.data), width, height);
  const magnitudeOffset = magnitude * 255;
  for (let index = 0; index < transformed.data.length; index += 4) {
    transformed.data[index] = Math.min(255, transformed.data[index] + magnitudeOffset);
    transformed.data[index + 1] = Math.min(255, transformed.data[index + 1] + magnitudeOffset);
    transformed.data[index + 2] = Math.min(255, transformed.data[index + 2] + magnitudeOffset);
  }
  context.putImageData(transformed, 0, 0);
}

function renderSpectrogram() {
  const frameCount = randomSegments.length
    ? randomSegments.reduce((total, segment) => total + segment.values.length, 0)
    : spectrogram.length;
  const frequencyCount = randomSegments[0]?.values[0]?.length || spectrogram[0]?.length || 0;
  const chartWidth = Math.max(waveformScroll.clientWidth, frameCount * PIXELS_PER_FRAME);
  const chartHeight = canvas.clientHeight || 270;
  const ratio = window.devicePixelRatio || 1;
  canvas.style.width = `${chartWidth}px`;
  canvas.width = chartWidth * ratio;
  canvas.height = chartHeight * ratio;
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, chartWidth, chartHeight);
  if (!frequencyCount) return;

  const image = context.createImageData(chartWidth, chartHeight);
  for (let x = 0; x < chartWidth; x += 1) {
    let frame;
    let stackSegment = null;
    if (randomSegments.length) {
      const frameIndex = Math.min(
        randomStackSpectrogram.length - 1,
        Math.floor(x / chartWidth * randomStackSpectrogram.length),
      );
      frame = randomStackSpectrogram[frameIndex];
      const progress = x / chartWidth;
      const totalDuration = randomSegments.reduce((total, item) => total + item.duration, 0);
      let elapsed = 0;
      for (const segment of randomSegments) {
        elapsed += segment.duration / totalDuration;
        if (progress < elapsed) {
          stackSegment = segment;
          break;
        }
      }
    } else {
      const frameIndex = Math.min(frameCount - 1, Math.floor(x / chartWidth * frameCount));
      frame = spectrogram[frameIndex];
    }
    for (let y = 0; y < chartHeight; y += 1) {
      const bin = Math.min(frequencyCount - 1, Math.floor((1 - y / chartHeight) * frequencyCount));
      const intensity = Math.min(1, Math.max(0, Math.pow(Number(frame?.[bin]) || 0, 0.42)));
      const offset = (y * chartWidth + x) * 4;
      const color = stackSegment?.color || [8, 31, 55];
      const heat = stackSegment ? intensity : intensity * 0.9;
      image.data[offset] = Math.round(color[0] + heat * (255 - color[0]));
      image.data[offset + 1] = Math.round(color[1] + heat * (220 - color[1]));
      image.data[offset + 2] = Math.round(color[2] + heat * (150 - color[2]));
      image.data[offset + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);

  if (playback && audioContext) {
    const progress = Math.min(1, Math.max(0, (audioContext.currentTime - playback.startedAt) / playback.duration));
      const playedX = chartWidth * progress;
      const viewportWidth = waveformScroll.clientWidth;
      const maxScroll = Math.max(0, chartWidth - viewportWidth);
      waveformScroll.scrollLeft = Math.min(maxScroll, Math.max(0, playedX - viewportWidth * 0.35));
      context.fillStyle = "rgb(255 126 38 / 0.42)";
      context.fillRect(0, 0, chartWidth * progress, chartHeight);
  }

  if (randomSegments.length) {
    let elapsed = 0;
    randomSegments.forEach((segment, index) => {
      const totalDuration = randomSegments.reduce((total, item) => total + item.duration, 0);
      const x = chartWidth * elapsed / totalDuration;
      elapsed += segment.duration;
      const endX = chartWidth * elapsed / totalDuration;
      context.strokeStyle = index === activeRandomSegment ? "#ffffff" : "rgb(255 255 255 / 0.35)";
      context.lineWidth = index === activeRandomSegment ? 3 : 1;
      context.strokeRect(x, 0, Math.max(1, endX - x), chartHeight);
    });
  }

  context.strokeStyle = `rgb(247 200 115 / ${Number(pitchBrightnessSlider.value) / 100})`;
  context.lineWidth = 2;
  context.beginPath();
  pitches.forEach((pitch, index) => {
    const x = index / Math.max(1, pitches.length - 1) * chartWidth;
    const y = chartHeight * (1 - Math.min(1, pitch / (metadata.sample_rate / 2)));
    index ? context.lineTo(x, y) : context.moveTo(x, y);
  });
  context.stroke();

  if (playback && audioContext) {
    const progress = Math.min(1, Math.max(0, (audioContext.currentTime - playback.startedAt) / playback.duration));
    const playedX = chartWidth * progress;
    context.strokeStyle = "#ffffff";
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(playedX, 0);
    context.lineTo(playedX, chartHeight);
    context.stroke();
  }
}

function animatePlayback(timestamp) {
  if (!playback) return;
  if (timestamp - playback.lastPaint >= 100) {
    playback.lastPaint = timestamp;
    if (playback.random) {
      const elapsed = audioContext.currentTime - playback.startedAt;
      let accumulated = 0;
      activeRandomSegment = randomSegments.findIndex((segment) => {
        accumulated += segment.duration;
        return elapsed < accumulated;
      });
    }
    renderSpectrogram();
    renderImageCanvas(Math.min(1, (audioContext.currentTime - playback.startedAt) / playback.duration));
  }
  if (audioContext.currentTime - playback.startedAt < playback.duration) {
    playback.animationFrame = requestAnimationFrame(animatePlayback);
  }
}

async function refreshSpectrogram() {
  const body = await requestJson("/api/spectrogram", { headers: { "X-Audio-Session": sessionId } });
  metadata = body;
  spectrogram = body.values;
  pitches = body.pitches;
  pitchMagnitudes = body.pitch_magnitudes;
  randomSegments = [];
  randomStackSpectrogram = [];
  activeRandomSegment = -1;
  rateSlider.value = body.sample_rate;
  if (metadataEl) {
    metadataEl.textContent = `${body.filename} · ${body.channels} channel(s) · ${body.sample_rate} Hz · ${body.duration.toFixed(2)} seconds`;
  }
  updateLabels();
  renderSpectrogram();
}

async function uploadImage() {
  if (!sessionId) return setError("Upload audio before loading an image.");
  if (!imageFileInput.files.length) return setError("Choose an image file first.");
  setError("");
  setStatus("Loading image...");
  const form = new FormData();
  form.append("image", imageFileInput.files[0]);
  try {
    const body = await requestJson("/api/image", { method: "POST", headers: { "X-Audio-Session": sessionId }, body: form });
    const image = new Image();
    image.onload = () => {
      imageCanvas.width = image.width;
      imageCanvas.height = image.height;
      imageCanvas.getContext("2d").drawImage(image, 0, 0);
      sourceImageData = imageCanvas.getContext("2d").getImageData(0, 0, image.width, image.height);
      renderImageCanvas();
    };
    image.src = body.image;
    imageNameEl.textContent = `${body.filename} · ${body.width} x ${body.height}`;
    setStatus("Image loaded");
  } catch (error) {
    setError(error.message);
    setStatus("Error");
  }
}

async function upload() {
  if (!fileInput.files.length) return setError("Choose a WAV or MP3 file first.");
  setError("");
  setStatus("Uploading...");
  const form = new FormData();
  form.append("audio", fileInput.files[0]);
  try {
    const response = await fetch("/api/upload", { method: "POST", body: form });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error);
    sessionId = body.session_id;
    await refreshSpectrogram();
    [saveButton, breakTensorButton, playButton, downloadButton, imageUploadButton, breakDurationInput].forEach((button) => { button.disabled = false; });
    setStatus("Loaded");
  } catch (error) {
    setError(error.message);
    setStatus("Error");
  }
}

async function saveChanges() {
  setError("");
  try {
    await requestJson("/api/apply", {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({ gain: Number(gainSlider.value), pitch: 0 }),
    });
    await requestJson("/api/resample", {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({ sample_rate: Number(rateSlider.value) }),
    });
    gainSlider.value = "1";
    await refreshSpectrogram();
    setStatus("Changes kept");
  } catch (error) {
    setError(error.message);
    setStatus("Error");
  }
}

async function breakTensor() {
  setError("");
  try {
    const body = await requestJson("/api/break-tensor", {
      method: "POST",
      headers: headers(),
      body: JSON.stringify({ seconds: Number(breakDurationInput.value) }),
    });
    breakDurationInput.value = body.chunk_seconds;
    breakDurationInput.disabled = false;
    randomTensorButton.disabled = false;
    playRandomButton.disabled = true;
    randomSegments = [];
    renderSpectrogram();
    setStatus(`Original tensor broken into ${body.tensor_count} ${body.chunk_seconds}-second tensors`);
  } catch (error) {
    setError(error.message);
    setStatus("Error");
  }
}

async function createRandomTensor() {
  setError("");
  try {
    const body = await requestJson("/api/random-tensor", { method: "POST", headers: headers() });
    randomSegments = body.segments.map((segment, index) => ({
      ...segment,
      color: [[20, 90, 180], [170, 70, 40], [35, 135, 85], [140, 70, 170], [180, 125, 30]][index % 5],
    }));
    randomStackSpectrogram = body.stack_spectrogram.values;
    spectrogram = randomStackSpectrogram;
    pitches = body.stack_spectrogram.pitches;
    pitchMagnitudes = body.stack_spectrogram.pitch_magnitudes;
    renderSpectrogram();
    playRandomButton.disabled = false;
    setStatus(`Random stack has ${body.stack_count} tensors (${body.random_duration.toFixed(2)} seconds)`);
  } catch (error) {
    setError(error.message);
    setStatus("Error");
  }
}

async function playAudio(tensor = "original") {
  try {
    const response = await fetch(`/api/download?tensor=${tensor}`, { headers: { "X-Audio-Session": sessionId } });
    if (!response.ok) throw new Error("Could not load the audio.");
    audioContext = audioContext || new AudioContext();
    await audioContext.resume();
    const buffer = await audioContext.decodeAudioData(await response.arrayBuffer());
    if (playback) playback.source.stop();
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);
    playback = {
      source,
      startedAt: audioContext.currentTime,
      duration: buffer.duration,
      lastPaint: 0,
      random: tensor === "random",
    };
    playback.animationFrame = requestAnimationFrame(animatePlayback);
    source.onended = () => {
      if (playback?.source !== source) return;
      playback = null;
      activeRandomSegment = -1;
      renderSpectrogram();
      setStatus("Loaded");
    };
    source.start();
    setStatus("Playing");
  } catch (error) {
    setError(error.message);
  }
}

async function downloadStacks() {
  setError("");
  try {
    const response = await fetch("/api/download?tensor=stacks", {
      headers: { "X-Audio-Session": sessionId },
    });
    if (!response.ok) {
      const body = await response.json();
      throw new Error(body.error || "Could not download the tensor stacks.");
    }
    const link = document.createElement("a");
    const objectUrl = URL.createObjectURL(await response.blob());
    link.href = objectUrl;
    link.download = "tensor_stacks.zip";
    link.click();
    setTimeout(() => URL.revokeObjectURL(objectUrl), 0);
    setStatus("Tensor stacks downloaded");
  } catch (error) {
    setError(error.message);
  }
}

fileInput.addEventListener("change", () => { fileNameEl.textContent = fileInput.files[0]?.name || "No file selected"; });
imageFileInput.addEventListener("change", () => { imageNameEl.textContent = imageFileInput.files[0]?.name || "Choose an image"; });
uploadButton.addEventListener("click", upload);
imageUploadButton.addEventListener("click", uploadImage);
gainSlider.addEventListener("input", updateLabels);
pitchBrightnessSlider.addEventListener("input", () => {
  updateLabels();
  renderSpectrogram();
});
rateSlider.addEventListener("input", updateLabels);
saveButton.addEventListener("click", saveChanges);
playButton.addEventListener("click", playAudio);
breakTensorButton.addEventListener("click", breakTensor);
randomTensorButton.addEventListener("click", createRandomTensor);
playRandomButton.addEventListener("click", () => playAudio("random"));
downloadButton.addEventListener("click", downloadStacks);
window.addEventListener("resize", renderSpectrogram);