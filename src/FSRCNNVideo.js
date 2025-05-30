import React, { useRef, useState } from "react";
import * as ort from "onnxruntime-web";

const FSRCNNVideo = () => {
  const [processing, setProcessing] = useState(false);
  const [originalUrl, setOriginalUrl] = useState(null);
  const [frames, setFrames] = useState([]);
  const [upscaledFrames, setUpscaledFrames] = useState([]);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [scaleFactor, setScaleFactor] = useState(2);

  const videoRef = useRef();
  const canvasRef = useRef();

  const handleVideoUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setOriginalUrl(url);
      setFrames([]);
      setUpscaledFrames([]);
    }
  };

  // Utility: Extract frame as ImageData from <video>
  const getFrame = (video, w, h) => {
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, w, h);
    return ctx.getImageData(0, 0, w, h);
  };

  // Core function: Upscale video frame-by-frame
  const processVideo = async () => {
    setProcessing(true);
    setUpscaledFrames([]);
    const video = videoRef.current;
    const duration = video.duration;
    const fps = 15; // you can change this
    const totalFrames = Math.floor(duration * fps);
    const frameInterval = 1 / fps;

    // Load ONNX model once
    ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create('/models/fsrcnn_x2.onnx');

    const w = video.videoWidth;
    const h = video.videoHeight;

    let upscaled = [];

    for (let i = 0; i < totalFrames; i++) {
      video.currentTime = i * frameInterval;
      await new Promise((resolve) => (video.onseeked = resolve));
      const imageData = getFrame(video, w, h);
      // 1. Extract Y, U, V (reuse your image code)
      const { tensorY, uPlane, vPlane, width, height } = extractYUV(imageData);

      // 2. Neural upscaling on Y
      const outputs = await session.run({ input: tensorY });
      const outputTensor = outputs.output || Object.values(outputs)[0];
      const outW = outputTensor.dims[3];
      const outH = outputTensor.dims[2];
      const upY = outputTensor.data;

      // 3. Upscale U and V
      const upU = upscaleChannelToSize(uPlane, width, height, outW, outH);
      const upV = upscaleChannelToSize(vPlane, width, height, outW, outH);

      // 4. Merge YUV and store RGB
      const rgbFrame = yuvMerge(upY, upU, upV, outW, outH);
      upscaled.push(rgbFrame);
      setUpscaledFrames([...upscaled]); // for preview as it runs
    }
    setProcessing(false);
  };

  // Copy from your previous code!
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
    // returns canvas.toDataURL
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

  // Simple playback of upscaled frames
  React.useEffect(() => {
    if (upscaledFrames.length === 0) return;
    let i = 0;
    const interval = setInterval(() => {
      setCurrentFrame(i);
      i++;
      if (i >= upscaledFrames.length) clearInterval(interval);
    }, 1000 / 15);
    return () => clearInterval(interval);
  }, [upscaledFrames]);

  return (
    <div>
      <h1>FSRCNN Video Super-Resolution</h1>
      <input type="file" accept="video/*" onChange={handleVideoUpload} />
      {originalUrl && (
        <div style={{ margin: 12 }}>
          <video
            ref={videoRef}
            src={originalUrl}
            controls
            width={320}
            style={{ display: "block" }}
          />
          <button onClick={processVideo} disabled={processing}>
            {processing ? "Processing..." : "Upscale Video"}
          </button>
        </div>
      )}

      {upscaledFrames.length > 0 && (
        <div>
          <h3>Upscaled Video (frame-by-frame)</h3>
          <img
            src={upscaledFrames[currentFrame]}
            alt="upscaled-frame"
            width={320 * scaleFactor}
            height={180 * scaleFactor}
            style={{ border: "1px solid #ccc", background: "#222" }}
          />
          <p>
            Frame {currentFrame + 1}/{upscaledFrames.length}
          </p>
        </div>
      )}
    </div>
  );
};

export default FSRCNNVideo;
