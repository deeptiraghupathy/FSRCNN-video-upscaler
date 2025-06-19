import React, { useRef, useState } from "react";
import * as ort from "onnxruntime-web";

const FSRCNNVideoCompareSync = () => {
  const [originalUrl, setOriginalUrl] = useState(null);
  const [frameMeta, setFrameMeta] = useState({ w: 0, h: 0, duration: 0 });
  const [videoReady, setVideoReady] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [playingSync, setPlayingSync] = useState(false);
  const [syncFps, setSyncFps] = useState(24);
  const [frameIdx, setFrameIdx] = useState(0);

  // Always use 2x (FSRCNN x2 model)
  const scaleFactor = 2;

  // URLs for showing the current frame in each panel
  const [classicFrameUrl, setClassicFrameUrl] = useState(null);
  const [neuralFrameUrl, setNeuralFrameUrl] = useState(null);

  // Frame counts for metrics
  const [neuralFrameCount, setNeuralFrameCount] = useState(0);
  const [bicubicFrameCount, setBicubicFrameCount] = useState(0);

  // Profiling state
  const [avgOnnx, setAvgOnnx] = useState(0);
  const [avgPixel, setAvgPixel] = useState(0);
  const [avgPost, setAvgPost] = useState(0);
  const [avgTotal, setAvgTotal] = useState(0);

  // FPS Metrics
  const [fpsOriginal, setFpsOriginal] = useState(0);
  const [fpsNeural, setFpsNeural] = useState(0);

  const [isPaused, setIsPaused] = useState(false);

  // Refs for video, ONNX, timers, accumulators
  const videoRef = useRef();
  const sessionRef = useRef();
  const playFlag = useRef(false);
  const pauseFlag = useRef(false);

  // FPS calculation
  const originalFramesThisSec = useRef(0);
  const neuralFramesThisSec = useRef(0);
  const lastFpsUpdate = useRef(performance.now());

  // Averaging for profiling
  const avgWindow = 30;
  const avgBufOnnx = useRef([]);
  const avgBufPixel = useRef([]);
  const avgBufPost = useRef([]);
  const avgBufTotal = useRef([]);

  // --- Enable ONNX WebGPU backend if possible ---
  React.useEffect(() => {
    // Try to enable WebGPU. Fallback to WASM.
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
        console.log("ONNX model loaded (x2)");
      })
      .catch((err) => {
        console.error("Failed to load ONNX model:", err);
        alert("Could not load the ONNX model. Check the path and filename.");
      });
  }, []);

  // ----- YUV/RGB conversion -----
  function rgbToYuv(r, g, b) {
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    const u = -0.168736 * r - 0.331264 * g + 0.5 * b + 128;
    const v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128;
    return [y, u, v];
  }
  function yuvToRgb(y, u, v) {
    u = u - 128;
    v = v - 128;
    let r = y + 1.402 * v;
    let g = y - 0.344136 * u - 0.714136 * v;
    let b = y + 1.772 * u;
    r = Math.round(Math.min(255, Math.max(0, r)));
    g = Math.round(Math.min(255, Math.max(0, g)));
    b = Math.round(Math.min(255, Math.max(0, b)));
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
    const tensorY = new ort.Tensor('float32', yArr, [1, 1, height, width]);
    return { tensorY, uPlane: uArr, vPlane: vArr, width, height };
  };

  // ----- Classic Bicubic Upscale -----
  const classicUpscaleCanvas = (imgData) => {
    const srcW = imgData.width;
    const srcH = imgData.height;
    const dstW = srcW * scaleFactor;
    const dstH = srcH * scaleFactor;
    const srcCanvas = document.createElement("canvas");
    srcCanvas.width = srcW;
    srcCanvas.height = srcH;
    srcCanvas.getContext("2d").putImageData(imgData, 0, 0);

    const dstCanvas = document.createElement("canvas");
    dstCanvas.width = dstW;
    dstCanvas.height = dstH;
    const dstCtx = dstCanvas.getContext("2d");
    dstCtx.imageSmoothingEnabled = true;
    dstCtx.imageSmoothingQuality = "high";
    dstCtx.drawImage(srcCanvas, 0, 0, dstW, dstH);
    return dstCanvas.toDataURL();
  };

  // Nearest-neighbor upscale for U/V
  const upscaleChannelToSize = (channelData, srcW, srcH, dstW, dstH) => {
    const upArr = new Float32Array(dstW * dstH);
    for (let y = 0; y < dstH; y++) {
      const srcY = Math.floor(y * srcH / dstH);
      for (let x = 0; x < dstW; x++) {
        const srcX = Math.floor(x * srcW / dstW);
        upArr[y * dstW + x] = channelData[srcY * srcW + srcX];
      }
    }
    return upArr;
  };

// Fast YUV to RGB, returns unclamped values (only clamp at the very end)
function fastYuvToRgb(y, u, v) {
  u = u - 128;
  v = v - 128;
  let r = y + 1.402 * v;
  let g = y - 0.344136 * u - 0.714136 * v;
  let b = y + 1.772 * u;
  return [r, g, b];
}

// Highly optimized merge using typed arrays and minimal clamping
const yuvMerge = (upY, upU, upV, width, height) => {
  // Use Uint8ClampedArray for final buffer, very fast for putImageData
  const pixelCount = width * height;
  const buf = new Uint8ClampedArray(pixelCount * 4);

  for (let i = 0; i < pixelCount; ++i) {
    // Merge YUV to RGB (Y normalized, U/V already 0-255)
    // upY is [0..1], upU/upV are [0..255]
    const y = upY[i] * 255.0;
    const u = upU[i];
    const v = upV[i];
    const [r, g, b] = fastYuvToRgb(y, u, v);

    buf[i * 4 + 0] = r < 0 ? 0 : r > 255 ? 255 : r;
    buf[i * 4 + 1] = g < 0 ? 0 : g > 255 ? 255 : g;
    buf[i * 4 + 2] = b < 0 ? 0 : b > 255 ? 255 : b;
    buf[i * 4 + 3] = 255; // Opaque alpha
  }

  // Canvas operations outside the loop
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const imageData = new ImageData(buf, width, height);
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL();
};


  // ---- Synchronized Playback Logic ----
  const playSynchronized = async () => {
    if (!videoRef.current || !sessionRef.current || !videoReady) {
      alert("Video not ready or ONNX not loaded!");
      return;
    }
    setClassicFrameUrl(null);
    setNeuralFrameUrl(null);
    setFrameIdx(0);
    setNeuralFrameCount(0);
    setBicubicFrameCount(0);
    setProcessing(true);
    setPlayingSync(true);
    setIsPaused(false);

    setAvgOnnx(0); setAvgPixel(0); setAvgPost(0); setAvgTotal(0);
    avgBufOnnx.current = [];
    avgBufPixel.current = [];
    avgBufPost.current = [];
    avgBufTotal.current = [];

    setFpsOriginal(0);
    setFpsNeural(0);
    originalFramesThisSec.current = 0;
    neuralFramesThisSec.current = 0;
    lastFpsUpdate.current = performance.now();

    playFlag.current = true;
    pauseFlag.current = false;

    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
    ort.env.wasm.simd = true;

    const video = videoRef.current;
    const session = sessionRef.current;
    const fps = Math.max(12, Math.min(syncFps, 100));
    const duration = video.duration;
    const totalFrames = Math.floor(duration * fps);
    const frameInterval = 1 / fps;

    let frameIdx = 0;
    let videoStartTime = performance.now();

    // FPS update function (every 1s)
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

    async function processNeuralFrame(imgData, ctxW, ctxH) {
      // Pixel extraction/proc timing
      const t0 = performance.now();
      const { tensorY, uPlane, vPlane } = extractYUV(imgData);
      const t1 = performance.now();

      // ONNX inference timing
      const t2 = performance.now();
      const outputs = await session.run({ input: tensorY });
      const t3 = performance.now();

      // Output postprocessing timing
      const outputTensor = outputs.output || Object.values(outputs)[0];
      const outW = outputTensor.dims[3];
      const outH = outputTensor.dims[2];
      const upY = outputTensor.data;
      const upU = upscaleChannelToSize(uPlane, ctxW, ctxH, outW, outH);
      const upV = upscaleChannelToSize(vPlane, ctxW, ctxH, outW, outH);
      const t4 = performance.now();
      const dataUrl = yuvMerge(upY, upU, upV, outW, outH);
      const t5 = performance.now();

      // Update rolling averages
      avgBufPixel.current.push(t1 - t0);
      avgBufOnnx.current.push(t3 - t2);
      avgBufPost.current.push((t4 - t3) + (t5 - t4));
      avgBufTotal.current.push(t5 - t0);
      if (avgBufOnnx.current.length > avgWindow) {
        avgBufPixel.current.shift(); avgBufOnnx.current.shift(); avgBufPost.current.shift(); avgBufTotal.current.shift();
      }
      setAvgPixel(avgBufPixel.current.reduce((a, b) => a + b, 0) / avgBufPixel.current.length);
      setAvgOnnx(avgBufOnnx.current.reduce((a, b) => a + b, 0) / avgBufOnnx.current.length);
      setAvgPost(avgBufPost.current.reduce((a, b) => a + b, 0) / avgBufPost.current.length);
      setAvgTotal(avgBufTotal.current.reduce((a, b) => a + b, 0) / avgBufTotal.current.length);

      return dataUrl;
    }

    async function displayLoop() {
      if (!playFlag.current || frameIdx >= totalFrames) {
        setPlayingSync(false);
        setProcessing(false);
        return;
      }
      if (pauseFlag.current) {
        setProcessing(false);
        return;
      }

      const now = performance.now();
      const elapsed = (now - videoStartTime) / 1000;
      let shouldFrame = Math.floor(elapsed * fps);

      if (shouldFrame > frameIdx + 2) {
        frameIdx = shouldFrame;
      }

      setFrameIdx(frameIdx + 1);

      // Seek video
      video.currentTime = frameIdx * frameInterval;
      await new Promise((resolve) => (video.onseeked = resolve));
      const ctxW = video.videoWidth;
      const ctxH = video.videoHeight;

      // Pixel extraction + classic timing
      const t0 = performance.now();
      const canvas = document.createElement("canvas");
      canvas.width = ctxW;
      canvas.height = ctxH;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, ctxW, ctxH);
      const imgData = ctx.getImageData(0, 0, ctxW, ctxH);
      const t1 = performance.now();

      // Show bicubic upscale immediately (so UI never blocks)
// Show bicubic upscale immediately (so UI never blocks)
const classicUrl = classicUpscaleCanvas(imgData);
setClassicFrameUrl(classicUrl);
originalFramesThisSec.current += 1;

// Try neural upscaling (if fails, fall back to classic)
let neuralUrl = null;
try {
  neuralUrl = await Promise.race([
    processNeuralFrame(imgData, ctxW, ctxH),
    new Promise((_, reject) => setTimeout(() => reject("timeout"), 250)), // 250ms timeout
  ]);
} catch (err) {
  neuralUrl = null;
}

if (neuralUrl) {
  setNeuralFrameUrl(neuralUrl);
  setNeuralFrameCount((c) => c + 1);
  neuralFramesThisSec.current += 1;
} else {
  setNeuralFrameUrl(classicUrl); // fallback to bicubic
  setBicubicFrameCount((c) => c + 1);  // Only increment fallback counter when neural fails
}


      updateFpsMetrics();
      frameIdx++;
      setTimeout(displayLoop, Math.max(1, 1000 / fps - 8));
    }

    displayLoop();
  };

  // Pause/resume logic
  const handlePauseResume = () => {
    if (!playingSync) return;
    if (isPaused) {
      pauseFlag.current = false;
      setIsPaused(false);
      setProcessing(true);
      playSynchronized();
    } else {
      pauseFlag.current = true;
      setIsPaused(true);
      setProcessing(false);
    }
  };

  // ----------- UI and Handlers -------------

  const onLoadedMetadata = () => {
    const video = videoRef.current;
    if (video) {
      setVideoReady(true);
      setFrameMeta({
        w: video.videoWidth,
        h: video.videoHeight,
        duration: video.duration,
      });
    }
  };

  const handleVideoUpload = (e) => {
    setVideoReady(false);
    setClassicFrameUrl(null);
    setNeuralFrameUrl(null);
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
    }
  };

  // ----------- Render -------------

  return (
    <div>
      <h1>FSRCNN Synchronized Video Frame Comparison (2x)</h1>
      <input type="file" accept="video/*" onChange={handleVideoUpload} />
      <div style={{ margin: 12 }}>
        <label>
          Scale Factor: <b>2x</b>
        </label>
        &nbsp; FPS:{" "}
        <input
          type="number"
          value={syncFps}
          min={12}
          max={100}
          onChange={(e) => setSyncFps(Number(e.target.value))}
          disabled={processing || playingSync}
          style={{ width: 50 }}
        />
        &nbsp;
        <button
          onClick={playSynchronized}
          disabled={processing || !videoReady || playingSync}
        >
          {playingSync ? "Playing..." : "Play Synchronized"}
        </button>
        &nbsp;
        <button
          onClick={handlePauseResume}
          disabled={!playingSync}
          style={{
            marginLeft: 12,
            background: isPaused ? "#8c0" : "#005",
            color: "#fff",
            fontWeight: "bold",
            borderRadius: 4,
            padding: "0 12px",
            cursor: "pointer",
            opacity: playingSync ? 1 : 0.6,
          }}
        >
          {isPaused ? "Resume" : "Pause"}
        </button>
        <span style={{ marginLeft: 16 }}>
          {frameIdx > 0 &&
            frameMeta.duration &&
            `Frame ${frameIdx} / ${Math.floor(frameMeta.duration * syncFps)}`}
        </span>
      </div>

      <video
        ref={videoRef}
        src={originalUrl}
        style={{ display: "none" }}
        onLoadedMetadata={onLoadedMetadata}
      />

      <div style={{ display: "flex", gap: 32, alignItems: "flex-start", marginTop: 16 }}>
        <div>
          <h3>Bicubic Upscale (2x)</h3>
          {classicFrameUrl && (
            <img
              src={classicFrameUrl}
              alt="classic"
              width={frameMeta.w * scaleFactor}
              height={frameMeta.h * scaleFactor}
              style={{
                border: "2px solid #999",
                background: "#222",
                objectFit: "contain",
              }}
            />
          )}
        </div>
        <div>
          <h3>Neural Upscale (FSRCNN x2)</h3>
          {neuralFrameUrl && (
            <img
              src={neuralFrameUrl}
              alt="neural"
              width={frameMeta.w * scaleFactor}
              height={frameMeta.h * scaleFactor}
              style={{
                border: "2px solid #999",
                background: "#222",
                objectFit: "contain",
              }}
            />
          )}
        </div>
      </div>
      {processing && <div>Processing frame {frameIdx}...</div>}
      {/* METRICS */}
      <div
        style={{
          marginTop: 24,
          padding: 12,
          border: "1px solid #ccc",
          borderRadius: 8,
          background: "#f8f8f8",
          width: 390,
        }}
      >
        <b>Frame Metrics</b>
        <br />
        Neural upscaled: <span style={{ color: "#28a745", fontWeight: "bold" }}>{neuralFrameCount}</span>
        <br />
        Bicubic fallback: <span style={{ color: "#d9534f", fontWeight: "bold" }}>{bicubicFrameCount}</span>
        <br /><br />
        <b>Playback FPS</b>
        <br />
        Original (left): <span style={{ color: "#007bff", fontWeight: "bold" }}>{fpsOriginal}</span> fps
        <br />
        Neural (right): <span style={{ color: "#f39c12", fontWeight: "bold" }}>{fpsNeural}</span> fps
        <br /><br />
        <b>Performance Metrics (ms, avg last {avgWindow})</b>
        <br />
        ONNX inference: <b>{avgOnnx.toFixed(1)}</b>
        <br />
        Pixel extraction: <b>{avgPixel.toFixed(1)}</b>
        <br />
        Postprocess: <b>{avgPost.toFixed(1)}</b>
        <br />
        <span>Total/frame: <b>{avgTotal.toFixed(1)}</b></span>
      </div>
      <div style={{fontSize:12, color:'#777', marginTop:10}}>
        Backend: <b>{ort.env.webgpu && ort.env.webgpu.enabled ? 'WebGPU or WASM' : 'WASM'}</b> (Check your browser console for logs)
      </div>
    </div>
  );
};

export default FSRCNNVideoCompareSync;
