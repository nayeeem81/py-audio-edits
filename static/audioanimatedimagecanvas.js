
const fileInput = document.querySelector("#audio-file");
const uploadButton = document.querySelector("#upload-button");
const imageFileInput = document.querySelector("#image-file");
const imageUploadButton = document.querySelector("#image-upload-button");
const canvas = document.querySelector("#image-canvas");
const imageNameEl = document.querySelector("#image-name");
const audio = document.getElementById("audioPlayer");

// 1. Load your background image
const bgImage = new Image();
bgImage.src = ""; // Path to your canvas image

async function uploadImage() {
  if (!imageFileInput.files.length) return setError("Choose an image file first.");
  setError("");
  setStatus("Loading image...");
  const form = new FormData();
  form.append("image", imageFileInput.files[0]);
  try {
    const body = await requestJson("/api/image", { method: "POST", headers: { "X-Audio-Session": sessionId }, body: form });
    const bgImage = new Image();
    bgImage.onload = () => {
      canvas.width = bgImage.width;
      canvas.height = bgImage.height;
      canvas.getContext("2d").drawImage(bgImage, 0, 0);
      sourceImageData = canvas.getContext("2d").getImageData(0, 0, bgImage.width, bgImage.height);
    };
    
    bgImage.src = body.image;
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
  
    setStatus("Loaded");
  } catch (error) {
    setError(error.message);
    setStatus("Error");
  }
}

let audioCtx, analyser, source, dataArray, bufferLength;
let imageLoaded = false;

bgImage.onload = () => {
    // Match canvas dimensions to the image source size
    canvas.width = bgImage.width;
    canvas.height = bgImage.height;
    ctx.drawImage(bgImage, 0, 0);
    imageLoaded = true;
};

// 2. Initialize Audio Context on user play action
audio.addEventListener("play", () => {
   if (!audioCtx) {
       audioCtx = new (window.AudioContext || window.webkitAudioContext)();
       analyser = audioCtx.createAnalyser();

       //Connect audio element to analyzer
        source = audioCtx.createMediaElementSource(audio);
        source.connect(analyser);
        analyser.connect(audioCtx.destination);
        
        // FFT Size determines the frequency resolution (and number of slices)
        analyser.fftSize = 512; 
        bufferLength = analyser.frequencyBinCount;
        dataArray = new Uint8Array(bufferLength);
    }
    animate();
});

// 3. The Animation Loop (Breaks the image vertically)
function animate() {
    if (!imageLoaded) return;
    
    // Request next animation frame
    requestAnimationFrame(animate);
    
    // Grab the current real-time frequency/magnitude data
    analyser.getByteFrequencyData(dataArray);
    
    // Clear canvas frame
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Determine how wide each slice should be relative to canvas and data points
    const sliceWidth = canvas.width / bufferLength;
    
    for (let i = 0; i < bufferLength; i++) {
        // Normalize magnitude value (0.0 to 1.0)
        const magnitude = dataArray[i] / 255; 
        
        // Calculate dynamic physical displacement (Break effect intensity)
        const verticalShift = magnitude * 80; // Shakes image vertically by up to 80px
        
        // Target positioning variables
        const sourceX = i * sliceWidth;
        const sourceY = 0;
        const sourceWidth = sliceWidth;
        const sourceHeight = canvas.height;
        
        // We draw vertical columns extracted out of the image asset, 
        // shifting their rendering position based on frequency energy.
        ctx.drawImage(
            bgImage,
            sourceX, sourceY, sourceWidth, sourceHeight,           // Source crop box
            sourceX, sourceY + verticalShift, sourceWidth, sourceHeight // Destination draw box
        );
        
        // Optional: Overlay a visible indicator line showing the magnitude edge
        ctx.strokeStyle = `rgba(255, 255, 255, ${magnitude})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(sourceX, sourceY + verticalShift);
        ctx.lineTo(sourceX + sliceWidth, sourceY + verticalShift);
        ctx.stroke();
    }
}

