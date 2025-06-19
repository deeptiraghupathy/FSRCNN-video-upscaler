import React, { useRef, useState } from "react";
import * as ort from "onnxruntime-web";
import shaka from "shaka-player";

// --- PSNR implementation for Uint8ClampedArray buffers ---
function computePSNR(bufA, bufB) {
  // bufA, bufB: Uint8ClampedArray from ImageData (.data), length = width*height*4 (RGBA)
  if (!bufA || !bufB || bufA.length !== bufB.length) return 0;
  let mse = 0;
  let n = bufA.length / 4;
  for (let i = 0; i < bufA.length; i += 4) {
    // Compare R, G, B only
    for (let c = 0; c < 3; ++c) {
      let d = bufA[i + c] - bufB[i + c];
      mse += d * d;
    }
  }
  mse /= (n * 3); // average over all color channels
  if (mse === 0) return 99.0; // identical images
  const PIXEL_MAX = 255;
  return 10 * Math.log10((PIXEL_MAX * PIXEL_MAX) / mse);
}

function computeSSIM(imgA, imgB) {
  // imgA, imgB: Uint8ClampedArray (.data) or Float32Array, length = w*h*4
  // We only use the Y (luma) channel for speed.
  if (!imgA || !imgB || imgA.length !== imgB.length) return 0;
  const K1 = 0.01, K2 = 0.03, L = 255;
  const C1 = (K1 * L) ** 2;
  const C2 = (K2 * L) ** 2;

  let muX = 0, muY = 0, sigmaX = 0, sigmaY = 0, sigmaXY = 0, N = 0;

  for (let i = 0; i < imgA.length; i += 4) {
    const yA = 0.299 * imgA[i] + 0.587 * imgA[i + 1] + 0.114 * imgA[i + 2];
    const yB = 0.299 * imgB[i] + 0.587 * imgB[i + 1] + 0.114 * imgB[i + 2];
    muX += yA;
    muY += yB;
    N++;
  }
  muX /= N;
  muY /= N;
  for (let i = 0; i < imgA.length; i += 4) {
    const yA = 0.299 * imgA[i] + 0.587 * imgA[i + 1] + 0.114 * imgA[i + 2];
    const yB = 0.299 * imgB[i] + 0.587 * imgB[i + 1] + 0.114 * imgB[i + 2];
    sigmaX += (yA - muX) * (yA - muX);
    sigmaY += (yB - muY) * (yB - muY);
    sigmaXY += (yA - muX) * (yB - muY);
  }
  sigmaX /= N - 1;
  sigmaY /= N - 1;
  sigmaXY /= N - 1;
  const ssim = ((2 * muX * muY + C1) * (2 * sigmaXY + C2)) /
    ((muX ** 2 + muY ** 2 + C1) * (sigmaX + sigmaY + C2));
  return ssim;
}

const FSRCNNVideoCompareSync = () => {
  const [originalUrl, setOriginalUrl] = useState(null);
  const [frameMeta, setFrameMeta] = useState({ w: 0, h: 0, duration: 0 });
  const [videoReady, setVideoReady] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [playingSync, setPlayingSync] = useState(false);
  const [syncFps, setSyncFps] = useState(24);
  const [frameIdx, setFrameIdx] = useState(0);

  const scaleFactor = 2;
  const [upscaledSize, setUpscaledSize] = useState({ w: frameMeta.w * scaleFactor, h: frameMeta.h * scaleFactor });

  const [psnrBicubic, setPsnrBicubic] = useState(null);
const [psnrNeural, setPsnrNeural] = useState(null);

const [ssimBicubic, setSsimBicubic] = useState(null);
const [ssimNeural, setSsimNeural] = useState(null);

  const [compareMode, setCompareMode] = useState("compare"); // "compare", "upscaled", "sidebyside"


  const [neuralFrameCount, setNeuralFrameCount] = useState(0);
  const [bicubicFrameCount, setBicubicFrameCount] = useState(0);

  const [avgOnnx, setAvgOnnx] = useState(0);
  const [avgPixel, setAvgPixel] = useState(0);
  const [avgPost, setAvgPost] = useState(0);
  const [avgTotal, setAvgTotal] = useState(0);

  const [fpsOriginal, setFpsOriginal] = useState(0);
  const [fpsNeural, setFpsNeural] = useState(0);

  const [isPaused, setIsPaused] = useState(false);
  const [onnxBackend, setOnnxBackend] = useState("unknown");

  // New: PSNR metrics
  const [currPSNR, setCurrPSNR] = useState(0);
  const [avgPSNR, setAvgPSNR] = useState(0);
  const psnrBuf = useRef([]);

  const [sliderX, setSliderX] = useState(0.5); // range [0, 1]
  const sliderDragRef = useRef(false);

  // Refs
  const videoRef = useRef();
  const audioVideoRef = useRef();
  const sessionRef = useRef();
  const playFlag = useRef(false);
  const pauseFlag = useRef(false);
  const displayLoopRef = useRef(); 

  const shakaPlayerRef = useRef(null);

  const hrCanvasRef = useRef();

  const classicCanvasRef = useRef();
  const neuralCanvasRef = useRef();

  const originalFramesThisSec = useRef(0);
  const neuralFramesThisSec = useRef(0);
  const lastFpsUpdate = useRef(performance.now());

  const avgWindow = 30;
  const avgBufOnnx = useRef([]);
  const avgBufPixel = useRef([]);
  const avgBufPost = useRef([]);
  const avgBufTotal = useRef([]);

  const HERO_GRAD = "linear-gradient(120deg, #1c2340 0%, #232c52 100%)";
  const CARD_BG = "rgba(23,30,51,0.88)";
  const CARD_ACCENT = "#57e4ff";
  const PRIMARY = "#47a6ff";
  const GREEN = "#38ef7d";
  const PINK = "#eb2c5b";
  const SHADOW = "0 6px 36px 0 rgba(60,70,130,0.10), 0 2px 12px 0 rgba(80,80,120,0.08)";

  // --- MOUSE HANDLERS FOR THE VERTICAL SLIDER ---
  const handleSliderDown = () => {
    if (compareMode !== "compare") return;
    sliderDragRef.current = true;
    document.body.style.cursor = "ew-resize";
  };
  const handleSliderUp = () => {
    sliderDragRef.current = false;
    document.body.style.cursor = "";
  };
  const handleSliderMove = (e) => {
    if (!sliderDragRef.current || compareMode !== "compare") return;
    const boundingBox = document.getElementById("compare-canvas-container").getBoundingClientRect();
    let clientX = e.touches ? e.touches[0].clientX : e.clientX;
    let x = (clientX - boundingBox.left) / boundingBox.width;
    x = Math.max(0, Math.min(1, x));
    setSliderX(x);
  };

  React.useEffect(() => {
    if (frameMeta.w && frameMeta.h) {
      setUpscaledSize({ w: frameMeta.w * scaleFactor, h: frameMeta.h * scaleFactor });
    }
  }, [frameMeta.w, frameMeta.h, scaleFactor]);

  
  React.useEffect(() => {
    const move = (e) => handleSliderMove(e);
    const up = () => handleSliderUp();
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
    window.addEventListener("touchmove", move);
    window.addEventListener("touchend", up);
    return () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      window.removeEventListener("touchmove", move);
      window.removeEventListener("touchend", up);
    };
    // eslint-disable-next-line
  }, [compareMode]);

  // Merge canvas for slider mode
  React.useEffect(() => {
    if (compareMode !== "compare") return;
    const mergeCanvas = document.getElementById("compare-merge-canvas");
    const bicubicCanvas = classicCanvasRef.current;
    const neuralCanvas = neuralCanvasRef.current;
    if (!mergeCanvas || !bicubicCanvas || !neuralCanvas) return;
    const w = bicubicCanvas.width;
    const h = bicubicCanvas.height;
    if (w === 0 || h === 0) return;
    mergeCanvas.width = w;
    mergeCanvas.height = h;
    const ctx = mergeCanvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    // Left: bicubic, right: neural. SliderX sets the split
    const splitX = Math.round(w * sliderX);

    // Draw left portion from bicubic
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, splitX, h);
    ctx.clip();
    ctx.drawImage(bicubicCanvas, 0, 0, w, h);
    ctx.restore();

    // Draw right portion from neural
    ctx.save();
    ctx.beginPath();
    ctx.rect(splitX, 0, w - splitX, h);
    ctx.clip();
    ctx.drawImage(neuralCanvas, 0, 0, w, h);
    ctx.restore();

    // Draw slider bar
    ctx.save();
    ctx.beginPath();
    ctx.rect(splitX - 2, 0, 4, h);
    ctx.clip();
    ctx.fillStyle = "#000";
    ctx.globalAlpha = 0.8;
    ctx.fillRect(splitX - 2, 0, 2, h);
    ctx.restore();
    // eslint-disable-next-line
  }, [sliderX, frameIdx, frameMeta.w, frameMeta.h, processing, compareMode]);

  // Shaka player for adaptive streams (for completeness)
  React.useEffect(() => {
    shaka.polyfill.installAll();
    return () => {
      if (shakaPlayerRef.current) {
        shakaPlayerRef.current.destroy();
        shakaPlayerRef.current = null;
      }
    };
  }, []);

  // ONNX backend init
  React.useEffect(() => {
    if (navigator.gpu) {
      console.log("WebGPU is available!");
    }
    try {
      ort.env.webgpu = ort.env.webgpu || {};
      ort.env.webgpu.enabled = true;
      ort.env.wasm.proxy = false;
      console.log("Attempting ONNX WebGPU backend...");
    } catch (e) {
      console.log("Could not enable ONNX WebGPU, falling back to WASM");
    }
    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
    ort.env.wasm.simd = true;

    ort.InferenceSession.create("/models/fsrcnn_x2.onnx", { executionProviders: ["webgpu", "wasm"] })
      .then((session) => {
        sessionRef.current = session;
        let backendName = "unknown";
        if (session.sessionOptions?.executionProviders && session.sessionOptions.executionProviders.length > 0) {
          backendName = session.sessionOptions.executionProviders[0];
        } else if (ort.env.webgpu.enabled && navigator.gpu) {
          backendName = "webgpu";
        } else {
          backendName = "wasm";
        }
        setOnnxBackend(backendName);
        console.log(`ONNX model loaded (x2), backend: ${backendName}`);
      })
      .catch((err) => {
        console.error("Failed to load ONNX model:", err);
        alert("Could not load the ONNX model. Check the path and filename.");
      });
  }, []);

  // Add this function for shaka use

  // ----- YUV/RGB conversion -----
  function rgbToYuv(r, g, b) {
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    const u = -0.168736 * r - 0.331264 * g + 0.5 * b + 128;
    const v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128;
    return [y, u, v];
  }
  function fastYuvToRgb(y, u, v) {
    u = u - 128;
    v = v - 128;
    let r = y + 1.402 * v;
    let g = y - 0.344136 * u - 0.714136 * v;
    let b = y + 1.772 * u;
    return [r, g, b];
  }
  const extractYUV = (imageData) => {
    const { data, width, height } = imageData;
    const yArr = new Float32Array(width * height);
    const uArr = new Float32Array(width * height);
    const vArr = new Float32Array(width * height);
    for (let i = 0; i < width * height; i++) {
      const r = data[i * 4 + 0];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];
      const [y, u, v] = rgbToYuv(r, g, b);
      yArr[i] = y / 255.0;
      uArr[i] = u;
      vArr[i] = v;
    }

    const tensorY = new ort.Tensor("float32", yArr, [1, 1, height, width]);
    return { tensorY, uPlane: uArr, vPlane: vArr, width, height };
  };

  // ----- Classic Bicubic Upscale -----
  const classicUpscaleCanvas = (imgData, targetCanvas, dstW, dstH) => {
    const srcW = imgData.width;
    const srcH = imgData.height;
    const srcCanvas = document.createElement("canvas");
    srcCanvas.width = srcW;
    srcCanvas.height = srcH;
    srcCanvas.getContext("2d").putImageData(imgData, 0, 0);
    targetCanvas.width = dstW;
    targetCanvas.height = dstH;
    const ctx = targetCanvas.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.clearRect(0, 0, dstW, dstH);
    ctx.drawImage(srcCanvas, 0, 0, dstW, dstH);
  };  

  // Nearest-neighbor upscale for U/V
  const upscaleChannelToSize = (channelData, srcW, srcH, dstW, dstH) => {
    const upArr = new Float32Array(dstW * dstH);
    for (let y = 0; y < dstH; y++) {
      const srcY = Math.floor((y * srcH) / dstH);
      for (let x = 0; x < dstW; x++) {
        const srcX = Math.floor((x * srcW) / dstW);
        upArr[y * dstW + x] = channelData[srcY * srcW + srcX];
      }
    }
    return upArr;
  };

  // Write merged YUV directly to canvas

  function blendImageData(imageDataA, imageDataB, t) {
    if (!imageDataA || !imageDataB) return imageDataA || imageDataB;
    const dataA = imageDataA.data;
    const dataB = imageDataB.data;
    const blended = new Uint8ClampedArray(dataA.length);
    for (let i = 0; i < dataA.length; i++) {
      blended[i] = Math.round(dataA[i] * (1 - t) + dataB[i] * t);
    }
    return new ImageData(blended, imageDataA.width, imageDataA.height);
  }

  function drawImageDataToCanvas(imageData, canvas) {
    if (!imageData) return;
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    const ctx = canvas.getContext("2d");
    ctx.putImageData(imageData, 0, 0);
  }

  function updateFpsMetrics() {
    const now = performance.now();
    const dt = (now - lastFpsUpdate.current) / 1000;
    if (dt >= 1) {
      setFpsOriginal(Math.round(originalFramesThisSec.current / dt));
      setFpsNeural(Math.round(neuralFramesThisSec.current / dt));
      originalFramesThisSec.current = 0;
      neuralFramesThisSec.current = 0;
      lastFpsUpdate.current = now;
    }
  }

  // --- Synchronized Playback Logic ---
  const playSynchronized = async (resumeFromPause = false) => {
    if (!videoRef.current || !audioVideoRef.current || !sessionRef.current || !videoReady) {
      alert("Video not ready or ONNX not loaded!");
      return;
    }
    // --- Only reset everything if not resuming from pause ---
  if (!resumeFromPause) {
    setFrameIdx(0);
    setNeuralFrameCount(0);
    setBicubicFrameCount(0);
    setProcessing(true);
    setPlayingSync(true);
    setIsPaused(false);

    setAvgOnnx(0);
    setAvgPixel(0);
    setAvgPost(0);
    setAvgTotal(0);
    avgBufOnnx.current = [];
    avgBufPixel.current = [];
    avgBufPost.current = [];
    avgBufTotal.current = [];

    setFpsOriginal(0);
    setFpsNeural(0);
    originalFramesThisSec.current = 0;
    neuralFramesThisSec.current = 0;
    lastFpsUpdate.current = performance.now();

    // Reset PSNR metrics
    psnrBuf.current = [];
    setCurrPSNR(0);
    setAvgPSNR(0);

    playFlag.current = true;
    pauseFlag.current = false;

    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
    ort.env.wasm.simd = true;

    // Only reset currentTime if not resuming from pause!
    const video = videoRef.current;
    const audioVideo = audioVideoRef.current;
    video.pause();
    audioVideo.currentTime = 0;
    video.currentTime = 0;
    audioVideo.volume = 1.0;
    audioVideo.muted = false;
    audioVideo.play();
  } else {
    setProcessing(true);
    setIsPaused(false);
    playFlag.current = true;
    pauseFlag.current = false;
    audioVideoRef.current.play();
  }
    setFrameIdx(0);
    setNeuralFrameCount(0);
    setBicubicFrameCount(0);
    setProcessing(true);
    setPlayingSync(true);
    setIsPaused(false);

    setAvgOnnx(0);
    setAvgPixel(0);
    setAvgPost(0);
    setAvgTotal(0);
    avgBufOnnx.current = [];
    avgBufPixel.current = [];
    avgBufPost.current = [];
    avgBufTotal.current = [];

    setFpsOriginal(0);
    setFpsNeural(0);
    originalFramesThisSec.current = 0;
    neuralFramesThisSec.current = 0;
    lastFpsUpdate.current = performance.now();

    // Reset PSNR metrics
    psnrBuf.current = [];
    setCurrPSNR(0);
    setAvgPSNR(0);

    playFlag.current = true;
    pauseFlag.current = false;

    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
    ort.env.wasm.simd = true;

    const video = videoRef.current;
    const audioVideo = audioVideoRef.current;
    const session = sessionRef.current;
    const targetFps = Math.max(12, Math.min(syncFps, 100));
    const duration = audioVideo.duration;
    const totalFrames = Math.floor(duration * targetFps);

    let neuralFrameBuf = [];
    let lastNeuralTime = -1;

    video.pause();
    audioVideo.currentTime = 0;
    video.currentTime = 0;

    audioVideo.volume = 1.0;
    audioVideo.muted = false;
    audioVideo.play();

    async function processNeuralFrameAsImageData(imgData, ctxW, ctxH) {
      const t0 = performance.now();
      const { tensorY, uPlane, vPlane } = extractYUV(imgData);
      const t1 = performance.now();

      const t2 = performance.now();
      const outputs = await session.run({ input: tensorY });
      const t3 = performance.now();

      const outputTensor = outputs.output || Object.values(outputs)[0];
      const outW = outputTensor.dims[3];
      const outH = outputTensor.dims[2];
// Update upscaledSize (if changed)
if (
  (upscaledSize.w !== outW || upscaledSize.h !== outH) &&
  outW > 0 && outH > 0
) {
  setUpscaledSize({ w: outW, h: outH });
}

classicUpscaleCanvas(imgData, classicCanvasRef.current, outW, outH);
      
      const upY = outputTensor.data;
      const upU = upscaleChannelToSize(uPlane, ctxW, ctxH, outW, outH);
      const upV = upscaleChannelToSize(vPlane, ctxW, ctxH, outW, outH);
      const t4 = performance.now();

      const pixelCount = outW * outH;
      const buf = new Uint8ClampedArray(pixelCount * 4);
      for (let i = 0; i < pixelCount; ++i) {
        const y = upY[i] * 255.0;
        const u = upU[i];
        const v = upV[i];
        const [r, g, b] = fastYuvToRgb(y, u, v);
        buf[i * 4 + 0] = r < 0 ? 0 : r > 255 ? 255 : r;
        buf[i * 4 + 1] = g < 0 ? 0 : g > 255 ? 255 : g;
        buf[i * 4 + 2] = b < 0 ? 0 : b > 255 ? 255 : b;
        buf[i * 4 + 3] = 255;
      }
      const imageData = new ImageData(buf, outW, outH);

      const t5 = performance.now();
      avgBufPixel.current.push(t1 - t0);
      avgBufOnnx.current.push(t3 - t2);
      avgBufPost.current.push(t4 - t3 + (t5 - t4));
      avgBufTotal.current.push(t5 - t0);
      if (avgBufOnnx.current.length > avgWindow) {
        avgBufPixel.current.shift();
        avgBufOnnx.current.shift();
        avgBufPost.current.shift();
        avgBufTotal.current.shift();
      }
      setAvgPixel(avgBufPixel.current.reduce((a, b) => a + b, 0) / avgBufPixel.current.length);
      setAvgOnnx(avgBufOnnx.current.reduce((a, b) => a + b, 0) / avgBufOnnx.current.length);
      setAvgPost(avgBufPost.current.reduce((a, b) => a + b, 0) / avgBufPost.current.length);
      setAvgTotal(avgBufTotal.current.reduce((a, b) => a + b, 0) / avgBufTotal.current.length);

      return imageData;
    }

    let lastFrameIdx = -1;
    let rafId;
    const neuralFrameInterval = 1.0 / (avgOnnx > 1 ? 1000 / avgOnnx : 10);

    const displayLoop = async () => {
      if (!playFlag.current) {
        setPlayingSync(false);
        setProcessing(false);
        audioVideo.pause();
        return;
      }
      if (pauseFlag.current) {
        setProcessing(false);
        audioVideo.pause();
        return;
      }

      // Draw original HR frame to hrCanvas
const hrW = upscaledSize.w;
const hrH = upscaledSize.h;
const hrCanvas = hrCanvasRef.current;
hrCanvas.width = hrW;
hrCanvas.height = hrH;
const hrCtx = hrCanvas.getContext("2d");
hrCtx.drawImage(video, 0, 0, hrW, hrH);


      const ctxW = video.videoWidth;
      const ctxH = video.videoHeight;
      const currentTime = audioVideo.currentTime;
      const currFrameIdx = Math.floor(currentTime * targetFps);

      if (currFrameIdx >= totalFrames || currentTime >= duration) {
        setPlayingSync(false);
        setProcessing(false);
        audioVideo.pause();
        return;
      }

      if (currFrameIdx === lastFrameIdx) {
        rafId = requestAnimationFrame(displayLoop);
        return;
      }
      lastFrameIdx = currFrameIdx;
      setFrameIdx(currFrameIdx + 1);

      video.currentTime = currentTime;
      await new Promise((resolve) => (video.onseeked = resolve));
      const tmpCanvas = document.createElement("canvas");
      tmpCanvas.width = ctxW;
      tmpCanvas.height = ctxH;
      const ctx = tmpCanvas.getContext("2d");
      ctx.drawImage(video, 0, 0, ctxW, ctxH);
      const imgData = ctx.getImageData(0, 0, ctxW, ctxH);

      if (compareMode === "compare" || compareMode === "sidebyside") {
        originalFramesThisSec.current += 1;
      }

      // Neural output
      let interpImage = null;

      // Neural output buffer and interpolation (for slider & upscaled)
      if (
        neuralFrameBuf.length === 0 ||
        currentTime - lastNeuralTime > neuralFrameInterval * 0.8
      ) {
        let neuralImageData = null;
        try {
          neuralImageData = await Promise.race([
            processNeuralFrameAsImageData(imgData, ctxW, ctxH),
            new Promise((_, reject) => setTimeout(() => reject("timeout"), 350)),
          ]);
          neuralFrameBuf.push({ time: currentTime, imageData: neuralImageData });
          lastNeuralTime = currentTime;
          if (neuralFrameBuf.length > 3) neuralFrameBuf.shift();
          setNeuralFrameCount((c) => c + 1);
          neuralFramesThisSec.current += 1;
        } catch (err) {}
      }

      let prevFrame = null,
        nextFrame = null;
      for (let i = 0; i < neuralFrameBuf.length; i++) {
        if (neuralFrameBuf[i].time <= currentTime) prevFrame = neuralFrameBuf[i];
        if (neuralFrameBuf[i].time > currentTime) {
          nextFrame = neuralFrameBuf[i];
          break;
        }
      }
      if (!prevFrame && neuralFrameBuf.length > 0) prevFrame = neuralFrameBuf[0];
      if (!nextFrame && neuralFrameBuf.length > 1) nextFrame = neuralFrameBuf[neuralFrameBuf.length - 1];

      interpImage = prevFrame?.imageData;
      if (prevFrame && nextFrame && nextFrame.time !== prevFrame.time) {
        const t = (currentTime - prevFrame.time) / (nextFrame.time - prevFrame.time);
        interpImage = blendImageData(prevFrame.imageData, nextFrame.imageData, t);
      }

      if (compareMode === "compare" || compareMode === "sidebyside") {
        if (neuralCanvasRef.current && interpImage) {
          drawImageDataToCanvas(interpImage, neuralCanvasRef.current);
                    // === Compute PSNR if both outputs are ready ===
                    try {
                      const classicCtx = classicCanvasRef.current.getContext("2d", { willReadFrequently: true });
                      const neuralCtx  = neuralCanvasRef.current.getContext("2d", { willReadFrequently: true });
                      const hrCtx      = hrCanvasRef.current.getContext("2d", { willReadFrequently: true });
                      
                      const w = classicCanvasRef.current.width, h = classicCanvasRef.current.height;
                      const wn = neuralCanvasRef.current.width, hn = neuralCanvasRef.current.height;
                      
                      const hrData      = hrCtx.getImageData(0, 0, hrW, hrH).data;
                      const bicubicData = classicCtx.getImageData(0, 0, hrW, hrH).data;
                      const neuralData  = neuralCtx.getImageData(0, 0, hrW, hrH).data;
                      
                      const psnr_bicubic = computePSNR(hrData, bicubicData);
const psnr_neural  = computePSNR(hrData, neuralData);


setPsnrBicubic(psnr_bicubic);
setPsnrNeural(psnr_neural);

const ssim_bicubic = computeSSIM(hrData, bicubicData);
const ssim_neural = computeSSIM(hrData, neuralData);

setSsimBicubic(ssim_bicubic);
setSsimNeural(ssim_neural);
                      if (bicubicData && neuralData && bicubicData.length === neuralData.length && w === wn && h === hn) {
                        const psnr = computePSNR(bicubicData, neuralData);
                        setCurrPSNR(psnr);
                        psnrBuf.current.push(psnr);
                        setAvgPSNR(psnrBuf.current.reduce((a, b) => a + b, 0) / psnrBuf.current.length);
                      }
                    } catch (e) {
                      console.warn("PSNR error:", e);
                    }
        } else if (neuralCanvasRef.current && classicCanvasRef.current) {
          const ctxN = neuralCanvasRef.current.getContext("2d");
          ctxN.clearRect(0, 0, classicCanvasRef.current.width, classicCanvasRef.current.height);
          ctxN.drawImage(classicCanvasRef.current, 0, 0);
          setBicubicFrameCount((c) => c + 1);
        }
      } else if (compareMode === "upscaled") {
        if (neuralCanvasRef.current && interpImage) {
          drawImageDataToCanvas(interpImage, neuralCanvasRef.current);
        }
      }

      updateFpsMetrics();
      rafId = requestAnimationFrame(displayLoop);
    };
    displayLoopRef.current = displayLoop;
    displayLoop();

    return () => {
      playFlag.current = false;
      if (rafId) cancelAnimationFrame(rafId);
      audioVideo.pause();
    };
  };

  // Pause/resume logic
const handlePauseResume = () => {
  if (!playingSync) return;
  if (isPaused) {
    // RESUME
    pauseFlag.current = false;
    setIsPaused(false);
    setProcessing(true);
    audioVideoRef.current.play();
    // Instead of calling playSynchronized(), just call the loop from current position
    if (displayLoopRef.current) {
      displayLoopRef.current();
    }
  } else {
    // PAUSE
    pauseFlag.current = true;
    setIsPaused(true);
    setProcessing(false);
    audioVideoRef.current.pause();
  }
};


  // Robust onLoadedMetadata
  const onLoadedMetadata = React.useCallback(() => {
    const video = videoRef.current;
    if (video && video.videoWidth && video.videoHeight && video.duration) {
      setVideoReady(true);
      setFrameMeta({
        w: video.videoWidth,
        h: video.videoHeight,
        duration: video.duration,
      });
    } else {
      setTimeout(onLoadedMetadata, 400);
    }
    // eslint-disable-next-line
  }, []);

  // Main file upload handler
  const handleVideoUpload = (e) => {
    setVideoReady(false);
    setFrameIdx(0);
    setNeuralFrameCount(0);
    setBicubicFrameCount(0);
    setProcessing(false);
    setPlayingSync(false);
    setIsPaused(false);
    setFpsOriginal(0);
    setFpsNeural(0);
    setAvgOnnx(0);
    setAvgPixel(0);
    setAvgPost(0);
    setAvgTotal(0);
    playFlag.current = false;
    pauseFlag.current = false;
    avgBufOnnx.current = [];
    avgBufPixel.current = [];
    avgBufPost.current = [];
    avgBufTotal.current = [];

    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setOriginalUrl(url);

      const isAdaptive =
        file.name.endsWith(".m3u8") ||
        file.type === "application/vnd.apple.mpegurl" ||
        file.name.endsWith(".mpd") ||
        file.type === "application/dash+xml";

      if (isAdaptive) {
        alert("Cannot play local adaptive streams (.m3u8, .mpd) directly in browser. Please use a network URL.");
        return;
      } else {
        if (videoRef.current) videoRef.current.src = url;
        if (audioVideoRef.current) audioVideoRef.current.src = url;
      }
    }
  };

  // Hosted video options
const hostedVideos = [
  { label: "Youtube 144p", url: "https://fsrcnnvideoupscaler.netlify.app/testVideos/TestVideo_144p.mp4" }, 
  { label: "Astro Action 144p", url: "https://fsrcnnvideoupscaler.netlify.app/testVideos/action_144p.mp4" },
  { label: "Astro Animation 144p", url: "https://fsrcnnvideoupscaler.netlify.app/testVideos/animation_144p.mp4" },
  { label: "Astro Dance 144p", url: "https://fsrcnnvideoupscaler.netlify.app/testVideos/video_144p.mp4" },
  { label: "Astro Animation 288p", url: "https://fsrcnnvideoupscaler.netlify.app/testVideos/animation_288p.mp4" },
  { label: "Astro Dance 288p", url: "https://fsrcnnvideoupscaler.netlify.app/testVideos/video_256p.mp4" }
];

const handleHostedVideoSelect = (e) => {
  const url = e.target.value;
  if (!url) return;
  setVideoReady(false);
  setFrameIdx(0);
  setNeuralFrameCount(0);
  setBicubicFrameCount(0);
  setProcessing(false);
  setPlayingSync(false);
  setIsPaused(false);
  setFpsOriginal(0);
  setFpsNeural(0);
  setAvgOnnx(0);
  setAvgPixel(0);
  setAvgPost(0);
  setAvgTotal(0);
  playFlag.current = false;
  pauseFlag.current = false;
  avgBufOnnx.current = [];
  avgBufPixel.current = [];
  avgBufPost.current = [];
  avgBufTotal.current = [];

  setOriginalUrl(url);

  if (videoRef.current) videoRef.current.src = url;
  if (audioVideoRef.current) audioVideoRef.current.src = url;
};

  // ----------- Render -------------

  return (
    <div style={{
      minHeight: "100vh",
      background: HERO_GRAD,
      color: "#e7efff",
      fontFamily: "'Inter', 'Segoe UI', Arial, sans-serif",
      padding: 0,
      margin: 0,
      boxSizing: "border-box"
    }}>
      <div style={{
        maxWidth: "none",
        margin: "0px",
        borderRadius: 32,
        background: "rgba(20,25,42,0.95)",
        boxShadow: SHADOW,
        padding: "0 0 44px 0",
        overflow: "hidden",
        border: "1.5px solid #273158",
        backdropFilter: "blur(10px)"
      }}>
        {/* --- HERO TITLE --- */}
        <div style={{
          background: "linear-gradient(90deg, #242d53 60%, #233 100%)",
          padding: "38px 0 24px 0",
          textAlign: "center",
          borderBottom: "1.5px solid #253052",
          position: "relative"
        }}>
          <span style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 18
          }}>

            <span style={{
              fontWeight: 900,
              fontSize: 29,
              background: "linear-gradient(92deg, #7cfaff 30%, #4ea2f5 100%)",
              backgroundClip: "text",
              WebkitBackgroundClip: "text",
              color: "transparent",
              lineHeight: 1.15,
              letterSpacing: "-.03em"
            }}>
              FSRCNN Video Super-Resolution <span style={{fontWeight: 700, fontSize: 20, color: "#73e2fecc"}}></span>
            </span>
          </span>
          <div style={{
            color: "#7cfaff",
            marginTop: 14,
            fontWeight: 500,
            fontSize: 17,
            letterSpacing: ".01em"
          }}>
            Real-Time AI Upscaling, Bicubic Comparison, Metrics & Audio Sync
          </div>
        </div>
  
        {/* --- CONTROLS --- */}
        <div style={{
          display: "flex", flexWrap: "wrap", gap: 18, justifyContent: "center",
          margin: "36px 0 14px 0", alignItems: "center"
        }}>
          {/* Upload */}
          <label style={{
            background: "rgba(46,61,110,0.95)",
            color: "#78d6ff", fontWeight: 700,
            padding: "10px 26px", borderRadius: 12,
            border: "1.5px solid #47a6ff", cursor: "pointer",
            fontSize: 16, boxShadow: "0 2px 16px 0 rgba(60,130,255,0.07)"
          }}>
            <input type="file" accept="video/*,.ts,video/mp2t,application/vnd.apple.mpegurl,.m3u8"
              onChange={handleVideoUpload}
              style={{display:"none"}} />
            <span style={{ pointerEvents: "none" }}>Upload Video</span>
          </label>

          {/* Hosted Video Select */}
<select
  onChange={handleHostedVideoSelect}
  defaultValue=""
  disabled={processing || playingSync}
  style={{
    background: "#23304c",
    color: "#a7fffa",
    borderRadius: 10,
    border: "1.5px solid #47a6ff",
    fontWeight: 700,
    fontSize: 16,
    padding: "10px 17px",
    outline: "none",
    minWidth: 170,
    marginLeft: 8
  }}>
  <option value="" disabled>Select Hosted Video</option>
  {hostedVideos.map((vid, idx) =>
    <option key={idx} value={vid.url}>{vid.label}</option>
  )}
</select>

          <span style={{ fontSize: 15, color: "#a6b8db", fontWeight: 500, margin: "0 6px" }}>
            <span style={{ color: "#fff", fontWeight: 700 }}>2x</span> Upscale
          </span>
          <span style={{ fontSize: 15, color: "#a6b8db", fontWeight: 500, margin: "0 6px" }}>
            FPS:
            <input
              type="number"
              value={syncFps}
              min={24}
              max={100}
              onChange={(e) => setSyncFps(Number(e.target.value))}
              disabled={processing || playingSync}
              style={{
                width: 54, fontSize: 16, padding: "3px 8px", marginLeft: 5,
                border: "1.5px solid #273158", borderRadius: 7, background: "#181f36", color: "#8df",
                fontWeight: 700
              }}
            />
          </span>
          {/* Display Mode */}
          <span style={{ fontWeight: 500, color: "#47a6ff", fontSize: 15, marginLeft: 14 }}>Mode:</span>
          <select
            value={compareMode}
            onChange={e => setCompareMode(e.target.value)}
            disabled={processing || playingSync}
            style={{
              background: "#223048", color: "#3df0d9", borderRadius: 8,
              border: "1.5px solid #3df0d9", fontWeight: 700, fontSize: 15,
              padding: "7px 13px", outline: "none", minWidth: 112, marginLeft: 2
            }}>
            <option value="compare">Slider</option>
            <option value="sidebyside">Dual Canvases</option>
            <option value="upscaled">Upscaled Only</option>
          </select>
          {/* Play/Pause */}
          <button
            onClick={playSynchronized}
            disabled={processing || !videoReady || playingSync}
            style={{
              background: PRIMARY, color: "#fff", fontWeight: 800, fontSize: 16,
              border: "none", borderRadius: 11, padding: "11px 34px",
              boxShadow: "0 1.5px 12px 0 rgba(70,170,255,0.12)",
              cursor: (processing || !videoReady || playingSync) ? "not-allowed" : "pointer",
              opacity: (processing || !videoReady || playingSync) ? 0.55 : 1,
              marginLeft: 15, transition: "background .17s,opacity .16s"
            }}>
            {playingSync ? (
              <span>
                <span className="spin" style={{
                  display: "inline-block",
                  marginRight: 7, width: 17, height: 17,
                  border: "3px solid #7cfaff", borderTop: "3px solid #256fa5",
                  borderRadius: "50%", animation: "spin 1s linear infinite",
                  verticalAlign: "middle"
                }} />
                Playing...
              </span>
            ) : "▶ Play"}
            <style>{`@keyframes spin{100%{transform:rotate(360deg);}}`}</style>
          </button>
          <button
            onClick={handlePauseResume}
            disabled={!playingSync}
            style={{
              background: isPaused ? GREEN : "#23304c", color: "#fff", fontWeight: 800, fontSize: 16,
              border: "none", borderRadius: 11, padding: "11px 34px",
              marginLeft: 5, boxShadow: "0 1.5px 10px 0 rgba(70,170,255,0.10)",
              cursor: playingSync ? "pointer" : "not-allowed", opacity: playingSync ? 1 : 0.60,
              transition: "background .16s,opacity .14s"
            }}>
            {isPaused ? " ⏯ Resume" : " ⏯ Pause"}
          </button>
          <span style={{ fontWeight: 600, fontSize: 15, color: "#c9eeff", marginLeft: 18 }}>
            {frameIdx > 0 && frameMeta.duration &&
              `Frame ${frameIdx} / ${Math.floor(frameMeta.duration * syncFps)}`}
          </span>
        </div>
  
        {/* --- HIDDEN VIDEOS/CANVASES --- */}
        <video ref={videoRef} src={originalUrl} style={{ display: "none" }} muted onLoadedMetadata={onLoadedMetadata} />
        <video ref={audioVideoRef} src={originalUrl} style={{ display: "none" }} controls={false} preload="auto" />
        <canvas ref={hrCanvasRef} style={{ display: "none" }} />
        <canvas ref={classicCanvasRef} width={upscaledSize.w} height={upscaledSize.h} style={{ display: "none" }} />
        <canvas ref={neuralCanvasRef} width={upscaledSize.w} height={upscaledSize.h} style={{ display: "none" }} />
  
        {/* --- MAIN VISUAL COMPARISON --- */}
        <div style={{
          margin: "30px auto 18px",
          background: "rgba(25,35,58,0.92)",
          borderRadius: 19, boxShadow: "0 2px 24px 0 rgba(40,70,120,0.12)",
          padding: "32px 24px", maxWidth: 820, minHeight: 320
        }}>
          {/* Slider */}
          {compareMode === "compare" && (
            <div id="compare-canvas-container"
              style={{
                position: "relative", display: "inline-block", userSelect: "none", margin: "0 auto",
                borderRadius: 15, boxShadow: "0 2px 36px 0 rgba(60,220,255,0.06)",
                border: "2.5px solid #1936b8"
              }}
              onMouseDown={handleSliderDown}
              onTouchStart={handleSliderDown}
              onTouchMove={handleSliderMove}
            >
              <canvas
                id="compare-merge-canvas"
                width={upscaledSize.w}
                height={upscaledSize.h}
                style={{
                  width: upscaledSize.w,
                  height: upscaledSize.h,
                  borderRadius: 15,
                  background: "#181f2e",
                  objectFit: "contain",
                  display: "block",
                  cursor: sliderDragRef.current ? "ew-resize" : "pointer",
                  touchAction: "none"
                }}
              />
              {/* Handle */}
              <div style={{
                position: "absolute", left: `calc(${sliderX * 100}% - 14px)`, top: 0, width: 28, height: "100%",
                display: "flex", alignItems: "center", justifyContent: "center", zIndex: 12, pointerEvents: "none"
              }}>
                <div style={{
                  width: 0, height: "78%", borderLeft: `5px solid ${CARD_ACCENT}`,
                  borderRadius: 2, pointerEvents: "auto", boxShadow: "0 0 12px #4ef3ef44"
                }} />
              </div>
              {/* Labels */}
              <div style={{
                position: "absolute", left: 22, top: 16, color: "#fff", background: "#0ec7fbe0",
                padding: "4px 17px", borderRadius: 9, fontWeight: 800, fontSize: 16, letterSpacing: ".01em",
                boxShadow: "0 2px 8px #3df0d966"
              }}>Bicubic</div>
              <div style={{
                position: "absolute", right: 22, top: 16, color: "#fff", background: "#47a6ffdd",
                padding: "4px 17px", borderRadius: 9, fontWeight: 800, fontSize: 16, letterSpacing: ".01em",
                boxShadow: "0 2px 8px #47a6ff55"
              }}>Neural</div>
            </div>
          )}
          {compareMode === "sidebyside" && (
            <div style={{ display: "flex", gap: 36, alignItems: "flex-start", justifyContent: "center" }}>
              <div>
                <div style={{ textAlign: "center", color: "#4ef3ef", fontWeight: 800, fontSize: 16, marginBottom: 7 }}>Bicubic (2x)</div>
                <canvas ref={classicCanvasRef} width={upscaledSize.w} height={upscaledSize.h}
                  style={{
                    width: upscaledSize.w,
                    height: upscaledSize.h,
                    border: "2px solid #3df0d9", borderRadius: 13,
                    background: "#181f2e", objectFit: "contain", boxShadow: "0 1.5px 8px #48a5fd12"
                  }}
                />
              </div>
              <div>
                <div style={{ textAlign: "center", color: "#47a6ff", fontWeight: 800, fontSize: 16, marginBottom: 7 }}>Neural (FSRCNN x2)</div>
                <canvas ref={neuralCanvasRef} width={upscaledSize.w} height={upscaledSize.h}
                  style={{
                    width: upscaledSize.w,
                    height: upscaledSize.h,
                    border: "2px solid #47a6ff", borderRadius: 13,
                    background: "#181f2e", objectFit: "contain", boxShadow: "0 1.5px 8px #48a5fd22"
                  }}
                />
              </div>
            </div>
          )}
          {compareMode === "upscaled" && (
            <canvas ref={neuralCanvasRef} width={upscaledSize.w} height={upscaledSize.h}
              style={{
                width: upscaledSize.w,
                height: upscaledSize.h,
                border: "2px solid #3df0d9", borderRadius: 15,
                background: "#181f2e", objectFit: "contain", display: "block",
                margin: "0 auto", marginTop: 6, boxShadow: "0 1.5px 8px #3df0d9bb"
              }}
            />
          )}
        </div>
  
        {/* Processing spinner */}
        {processing && (
          <div style={{
            color: "#7cfaff", fontWeight: 700, fontSize: 18, margin: "18px auto 6px", textAlign: "center"
          }}>
            <span className="spin" style={{
              display: "inline-block",
              marginRight: 9,
              width: 20, height: 20,
              border: "4px solid #3df0d9", borderTop: "4px solid #23304c",
              borderRadius: "50%", animation: "spin 0.95s linear infinite",
              verticalAlign: "middle"
            }} />{" "}
            Processing frame <b style={{color:"#fff"}}>{frameIdx}</b>...
            <style>{`@keyframes spin{100%{transform:rotate(360deg);}}`}</style>
          </div>
        )}
  
        {/* --- METRICS --- */}
        <div style={{
          margin: "38px auto 0", display: "flex", flexWrap: "wrap", justifyContent: "center",
          gap: 26, maxWidth: 960
        }}>
          {/* Neural frames */}
          <MetricCard label="Neural Frames" color={GREEN} value={neuralFrameCount} />
          {(compareMode === "compare" || compareMode === "sidebyside") && (
            <MetricCard label="Bicubic Frames" color={PINK} value={bicubicFrameCount} />
          )}
          <MetricCard
            label="Neural FPS"
            value={fpsNeural}
            color="#47a6ff"
            unit="fps"
          />
          {(compareMode === "compare" || compareMode === "sidebyside") && (
            <MetricCard
              label="Original FPS"
              value={fpsOriginal}
              color="#3df0d9"
              unit="fps"
            />
          )}
          <MetricCard
            label={`ONNX (ms)`}
            value={avgOnnx.toFixed(1)}
            color="#7cfaff"
          />
          <MetricCard
            label={`Total/frame (ms)`}
            value={avgTotal.toFixed(1)}
            color="#fa7cd3"
          />
          <MetricCard
            label={`PSNR Neural vs HR`}
            value={psnrNeural !== null ? psnrNeural.toFixed(2) : "--"}
            color="#f9f947"
            unit="dB"
          />
          <MetricCard
            label={`PSNR Bicubic vs HR`}
            value={psnrBicubic !== null ? psnrBicubic.toFixed(2) : "--"}
            color="#ffbc48"
            unit="dB"
          />
          <MetricCard
            label="SSIM Neural vs HR"
            value={ssimNeural !== null ? ssimNeural.toFixed(4) : "--"}
            color="#7cfaff"
          />
          <MetricCard
            label="SSIM Bicubic vs HR"
            value={ssimBicubic !== null ? ssimBicubic.toFixed(4) : "--"}
            color="#c2fa7c"
          />
        </div>
  
        {/* Backend Info */}
        <div style={{
          fontSize: 15, color: "#93cfff", marginTop: 30, textAlign: "center", fontWeight: 700,
          letterSpacing: ".02em", paddingBottom: 6
        }}>
          Backend:&nbsp;
          <b style={{
            color: onnxBackend === "webgpu" ? "#23c662" : "#47a6ff",
            textTransform: "uppercase", fontWeight: 900, letterSpacing: ".04em"
          }}>{onnxBackend}</b>
          &nbsp; <span style={{color:"#7cfaff"}}>(see browser console for ONNX logs)</span>
        </div>
      </div>
    </div>
  );
  
  // --- MetricCard Component ---
  function MetricCard({ label, value, unit, color }) {
    return (
      <div style={{
        minWidth: 128, minHeight: 95, borderRadius: 17, background: CARD_BG,
        border: `2.5px solid ${color}`, boxShadow: "0 1.5px 12px 0 rgba(70,80,100,0.10)",
        margin: "0 0", padding: "15px 17px", textAlign: "center",
        display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center"
      }}>
        <div style={{
          fontSize: 19, color: color, fontWeight: 800, marginBottom: 6, letterSpacing: ".02em"
        }}>{label}</div>
        <div style={{
          fontSize: 30, fontWeight: 900, color: "#fff", margin: 0, letterSpacing: "-.03em"
        }}>
          {value} <span style={{fontSize:15, color:"#b8dfff"}}>{unit || ""}</span>
        </div>
      </div>
    );
  }
};

export default FSRCNNVideoCompareSync;
