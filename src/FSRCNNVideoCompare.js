import React, { useRef, useState } from "react";
import * as ort from "onnxruntime-web";

const FSRCNNVideoCompareSync = () => {
  const [originalUrl, setOriginalUrl] = useState(null);
  const [scaleFactor, setScaleFactor] = useState(2);
  const [frameMeta, setFrameMeta] = useState({ w: 0, h: 0, duration: 0 });
  const [videoReady, setVideoReady] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [playingSync, setPlayingSync] = useState(false);
  const [syncFps, setSyncFps] = useState(10);
  const [frameIdx, setFrameIdx] = useState(0);

  const [classicFrameUrl, setClassicFrameUrl] = useState(null);
  const [originalFrameUrl, setOriginalFrameUrl] = useState(null);
  const [neuralFrameUrl, setNeuralFrameUrl] = useState(null);

  const [leftMode, setLeftMode] = useState("classic"); // "classic" or "original"

  const videoRef = useRef();
  const sessionRef = useRef();
  const playFlag = useRef(false);

  // Load ONNX model once
  React.useEffect(() => {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;
    ort.env.wasm.proxy = true;
    ort.InferenceSession.create("/models/fsrcnn_x4.onnx")
      .then((session) => {
        sessionRef.current = session;
        console.log("ONNX model loaded");
      })
      .catch((err) => {
        console.error("Failed to load ONNX model:", err);
        alert("Could not load the ONNX model. Check the path and filename.");
      });
  }, []);

  // Frame extraction & processing helpers
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

  // Classic upscale with canvas
  const classicUpscaleCanvas = (imgData, scale) => {
    const srcW = imgData.width;
    const srcH = imgData.height;
    const dstW = srcW * scale;
    const dstH = srcH * scale;
    const srcCanvas = document.createElement("canvas");
    srcCanvas.width = srcW;
    srcCanvas.height = srcH;
    srcCanvas.getContext("2d").putImageData(imgData, 0, 0);

    const dstCanvas = document.createElement("canvas");
    dstCanvas.width = dstW;
    dstCanvas.height = dstH;
    const dstCtx = dstCanvas.getContext("2d");
    dstCtx.imageSmoothingEnabled = true;
    dstCtx.drawImage(srcCanvas, 0, 0, dstW, dstH);
    return dstCanvas.toDataURL();
  };

  // For "original" mode, just turn frame into PNG data URL
  const getImageDataUrl = (imgData) => {
    const srcW = imgData.width;
    const srcH = imgData.height;
    const srcCanvas = document.createElement("canvas");
    srcCanvas.width = srcW;
    srcCanvas.height = srcH;
    srcCanvas.getContext("2d").putImageData(imgData, 0, 0);
    return srcCanvas.toDataURL();
  };

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

  const yuvMerge = (upY, upU, upV, width, height) => {
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    const imageData = ctx.createImageData(width, height);
    for (let i = 0; i < width * height; i++) {
      const y = upY[i] * 255.0;
      const u = Math.max(0, Math.min(255, upU[i]));
      const v = Math.max(0, Math.min(255, upV[i]));
      const [r, g, b] = yuvToRgb(y, u, v);
      imageData.data[i * 4 + 0] = r;
      imageData.data[i * 4 + 1] = g;
      imageData.data[i * 4 + 2] = b;
      imageData.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL();
  };

  const playSynchronized = async () => {
    if (!videoRef.current || !sessionRef.current || !videoReady) {
      alert("Video not ready or ONNX not loaded!");
      return;
    }
    setClassicFrameUrl(null);
    setOriginalFrameUrl(null);
    setNeuralFrameUrl(null);
    setFrameIdx(0);
    setProcessing(true);
    setPlayingSync(true);
    playFlag.current = true;

    const video = videoRef.current;
    const session = sessionRef.current;
    const fps = syncFps;
    const duration = video.duration;
    const w = video.videoWidth;
    const h = video.videoHeight;
    const totalFrames = Math.floor(duration * fps);
    const frameInterval = 1 / fps;

    for (let i = 0; i < totalFrames && playFlag.current; i++) {
      setFrameIdx(i + 1);

      video.currentTime = i * frameInterval;
      await new Promise((resolve) => (video.onseeked = resolve));

      const ctxW = video.videoWidth;
      const ctxH = video.videoHeight;
      const canvas = document.createElement("canvas");
      canvas.width = ctxW;
      canvas.height = ctxH;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, ctxW, ctxH);
      const imgData = ctx.getImageData(0, 0, ctxW, ctxH);

      if (leftMode === "classic") {
        setClassicFrameUrl(classicUpscaleCanvas(imgData, scaleFactor));
        setOriginalFrameUrl(null);
      } else {
        setOriginalFrameUrl(getImageDataUrl(imgData));
        setClassicFrameUrl(null);
      }

      // --- Neural upscale ---
      try {
        const { tensorY, uPlane, vPlane, width, height } = extractYUV(imgData);
        const outputs = await session.run({ input: tensorY });
        const outputTensor = outputs.output || Object.values(outputs)[0];
        const outW = outputTensor.dims[3];
        const outH = outputTensor.dims[2];
        const upY = outputTensor.data;
        const upU = upscaleChannelToSize(uPlane, width, height, outW, outH);
        const upV = upscaleChannelToSize(vPlane, width, height, outW, outH);
        setNeuralFrameUrl(yuvMerge(upY, upU, upV, outW, outH));
      } catch (err) {
        console.error("ONNX inference error:", err);
        alert("Neural upscaling failed: " + err);
        setPlayingSync(false);
        setProcessing(false);
        return;
      }
    }
    setPlayingSync(false);
    setProcessing(false);
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
    setOriginalFrameUrl(null);
    setNeuralFrameUrl(null);
    setFrameIdx(0);
    playFlag.current = false;
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setOriginalUrl(url);
    }
  };

  // ----------- Render -------------

  return (
    <div>
      <h1>FSRCNN Synchronized Video Frame Comparison</h1>
      <input type="file" accept="video/*" onChange={handleVideoUpload} />
      <div style={{ margin: 12 }}>
        <label>
          Scale Factor:{" "}
          <select
            value={scaleFactor}
            onChange={(e) => setScaleFactor(Number(e.target.value))}
            disabled={processing || playingSync}
          >
            <option value="2">2x</option>
            <option value="3">3x</option>
            <option value="4">4x</option>
          </select>
        </label>
        &nbsp; FPS:{" "}
        <input
          type="number"
          value={syncFps}
          min={1}
          max={30}
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
        <span style={{ marginLeft: 16 }}>
          {frameIdx > 0 &&
            frameMeta.duration &&
            `Frame ${frameIdx} / ${Math.floor(frameMeta.duration * syncFps)}`}
        </span>
        <div style={{ marginLeft: 24, display: "inline-block" }}>
          <label>
            <input
              type="radio"
              name="leftmode"
              checked={leftMode === "classic"}
              disabled={processing || playingSync}
              onChange={() => setLeftMode("classic")}
              style={{ marginRight: 4 }}
            />
            Classic Upscale
          </label>
          &nbsp;&nbsp;
          <label>
            <input
              type="radio"
              name="leftmode"
              checked={leftMode === "original"}
              disabled={processing || playingSync}
              onChange={() => setLeftMode("original")}
              style={{ marginRight: 4 }}
            />
            Original (native size)
          </label>
        </div>
      </div>

      <video
        ref={videoRef}
        src={originalUrl}
        style={{ display: "none" }}
        onLoadedMetadata={onLoadedMetadata}
      />

      <div style={{ display: "flex", gap: 24, alignItems: "flex-start", marginTop: 16 }}>
        {/* Left: Classic Upscale or Original */}
        {leftMode === "classic" && classicFrameUrl && (
          <div>
            <h3>Classic Upscale</h3>
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
          </div>
        )}
        {leftMode === "original" && originalFrameUrl && (
          <div>
            <h3>Original (native)</h3>
            <img
              src={originalFrameUrl}
              alt="original"
              width={frameMeta.w}
              height={frameMeta.h}
              style={{
                border: "2px solid #999",
                background: "#222",
                objectFit: "contain",
              }}
            />
          </div>
        )}
        {/* Neural (right) */}
        {neuralFrameUrl && (
          <div>
            <h3>Neural Upscale (FSRCNN)</h3>
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
          </div>
        )}
      </div>
      {processing && <div>Processing frame {frameIdx}...</div>}
    </div>
  );
};

export default FSRCNNVideoCompareSync;
