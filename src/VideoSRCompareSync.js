import React, { useRef, useState } from "react";
import * as ort from "onnxruntime-web";

const VideoSRCompareSync = () => {
  const [originalUrl, setOriginalUrl] = useState(null);
  const [scaleFactor, setScaleFactor] = useState(2); // Both models must be x2
  const [frameMeta, setFrameMeta] = useState({ w: 0, h: 0, duration: 0 });
  const [videoReady, setVideoReady] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [playingSync, setPlayingSync] = useState(false);
  const [syncFps, setSyncFps] = useState(10);
  const [frameIdx, setFrameIdx] = useState(0);

  const [fsrcnnFrameUrl, setFSRCNNFrameUrl] = useState(null);
  const [espcnFrameUrl, setESPCNFrameUrl] = useState(null);

  const [paused, setPaused] = useState(false);
  const pausedRef = useRef(false);

  const videoRef = useRef();
  const fsrcnnSessionRef = useRef();
  const espcnSessionRef = useRef();
  const playFlag = useRef(false);

  // Load both ONNX models once
  React.useEffect(() => {
    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
    ort.env.wasm.simd = true;
    ort.env.wasm.proxy = true;

    ort.InferenceSession.create("/models/fsrcnn_x2.onnx")
      .then((session) => {
        fsrcnnSessionRef.current = session;
        console.log("FSRCNN ONNX loaded");
      })
      .catch((err) => {
        alert("Could not load FSRCNN ONNX model.");
      });

    ort.InferenceSession.create("/models/espcn_x2.onnx")
      .then((session) => {
        espcnSessionRef.current = session;
        console.log("ESPCN ONNX loaded");
      })
      .catch((err) => {
        alert("Could not load ESPCN ONNX model.");
      });
  }, []);

  const setPauseBoth = (val) => {
    setPaused(val);
    pausedRef.current = val;
  };
  
  // YUV extraction for FSRCNN
  function rgbToYuv(r, g, b) {
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    const u = -0.168736 * r - 0.331264 * g + 0.5 * b + 128;
    const v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128;
    return [y, u, v];
  }
  function yuvToRgb(y, u, v) {
    u -= 128;
    v -= 128;
    let r = y + 1.402 * v;
    let g = y - 0.344136 * u - 0.714136 * v;
    let b = y + 1.772 * u;
    return [
      Math.round(Math.min(255, Math.max(0, r))),
      Math.round(Math.min(255, Math.max(0, g))),
      Math.round(Math.min(255, Math.max(0, b))),
    ];
  }
  // Extract YUV for FSRCNN
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
    return { tensorY, uArr, vArr, width, height };
  };
  // Upscale U/V channels with browser
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
  // Merge YUV back to RGB
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

  // Extract RGB for ESPCN
  const extractRGB = (imageData) => {
    const { data, width, height } = imageData;
    const floatArr = new Float32Array(3 * width * height);
    for (let i = 0; i < width * height; i++) {
      floatArr[i] = data[i * 4] / 255.0; // R
      floatArr[width * height + i] = data[i * 4 + 1] / 255.0; // G
      floatArr[2 * width * height + i] = data[i * 4 + 2] / 255.0; // B
    }
    const tensor = new ort.Tensor("float32", floatArr, [1, 3, height, width]);
    return { tensor, width, height };
  };
  // ESPCN output tensor to ImageData
  const tensorToImageData = (tensor, outW, outH) => {
    const arr = tensor.data;
    const imageData = new ImageData(outW, outH);
    const size = outW * outH;
    for (let i = 0; i < size; i++) {
      imageData.data[i * 4] = Math.round(Math.min(255, Math.max(0, arr[i] * 255)));
      imageData.data[i * 4 + 1] = Math.round(Math.min(255, Math.max(0, arr[size + i] * 255)));
      imageData.data[i * 4 + 2] = Math.round(Math.min(255, Math.max(0, arr[2 * size + i] * 255)));
      imageData.data[i * 4 + 3] = 255;
    }
    return imageData;
  };

  // Classic upscale with canvas (optional for baseline, can be removed)
  // const classicUpscaleCanvas = ...

  // Play video and show FSRCNN/ESPCN side by side
  const playSynchronized = async () => {
    setPaused(false);
    pausedRef.current = false;
    

    if (!videoRef.current || !fsrcnnSessionRef.current || !espcnSessionRef.current || !videoReady) {
      alert("Video or ONNX models not loaded!");
      return;
    }
    setFSRCNNFrameUrl(null);
    setESPCNFrameUrl(null);
    setFrameIdx(0);
    setProcessing(true);
    setPlayingSync(true);
    playFlag.current = true;

    const video = videoRef.current;
    const fsrcnnSession = fsrcnnSessionRef.current;
    const espcnSession = espcnSessionRef.current;
    const fps = syncFps;
    const duration = video.duration;
    const totalFrames = Math.floor(duration * fps);
    const frameInterval = 1 / fps;
    let startWallTime = performance.now();
    let i = 0;

    async function processFrame() {
      const elapsedSec = (performance.now() - startWallTime) / 1000;
      let expectedFrame = Math.floor(elapsedSec * fps);
      if (expectedFrame > i) i = expectedFrame;
      if (i >= totalFrames || !playFlag.current) {
        setPlayingSync(false);
        setProcessing(false);
        return;
      }
      setFrameIdx(i + 1);
      video.currentTime = i * frameInterval;
      await new Promise((resolve) => (video.onseeked = resolve));
      const ctxW = video.videoWidth;
      const ctxH = video.videoHeight;
      // --- Prepare input for both models ---
      const canvas = document.createElement("canvas");
      canvas.width = ctxW;
      canvas.height = ctxH;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, ctxW, ctxH);
      const imgData = ctx.getImageData(0, 0, ctxW, ctxH);

      // --- FSRCNN ---
      try {
        const { tensorY, uArr, vArr, width, height } = extractYUV(imgData);
        const outputs = await fsrcnnSession.run({ input: tensorY });
        const outputTensor = outputs.output || Object.values(outputs)[0];
        const outW = outputTensor.dims[3];
        const outH = outputTensor.dims[2];
        const upY = outputTensor.data;
        const upU = upscaleChannelToSize(uArr, width, height, outW, outH);
        const upV = upscaleChannelToSize(vArr, width, height, outW, outH);
        setFSRCNNFrameUrl(yuvMerge(upY, upU, upV, outW, outH));
      } catch (err) {
        setFSRCNNFrameUrl(null);
      }

      // --- ESPCN ---
      try {
        const { tensor } = extractRGB(imgData);
        const outputs = await espcnSession.run({ input: tensor });
        const outputTensor = outputs.output || Object.values(outputs)[0];
        const outW = outputTensor.dims[3];
        const outH = outputTensor.dims[2];
        const upscaledImageData = tensorToImageData(outputTensor, outW, outH);
        const upCanvas = document.createElement("canvas");
        upCanvas.width = outW;
        upCanvas.height = outH;
        upCanvas.getContext("2d").putImageData(upscaledImageData, 0, 0);
        setESPCNFrameUrl(upCanvas.toDataURL());
      } catch (err) {
        setESPCNFrameUrl(null);
      }

      i++;
      if (playFlag.current && !pausedRef.current) {
        setTimeout(processFrame, 0);
      } else if (playFlag.current && pausedRef.current) {
        const waitUntilResume = () => {
          if (!pausedRef.current && playFlag.current) {
            processFrame();
          } else if (playFlag.current) {
            setTimeout(waitUntilResume, 100);
          }
        };
        setTimeout(waitUntilResume, 100);
      }      
      
    }

    processFrame();
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
    setFSRCNNFrameUrl(null);
    setESPCNFrameUrl(null);
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
        <div>
  processing: {processing.toString()} <br/>
  playingSync: {playingSync.toString()} <br/>
  videoReady: {videoReady.toString()}
</div>

      <h1>FSRCNN vs ESPCN Video Upscaling Comparison</h1>
      <input type="file" accept="video/*" onChange={handleVideoUpload} />
      <div style={{ margin: 12 }}>
        Scale Factor: <b>2x</b>
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
        &nbsp;
<button
  onClick={() => setPauseBoth(true)}
  disabled={!playingSync || paused}
>
  Pause
</button>
&nbsp;
<button
  onClick={() => setPauseBoth(false)}
  disabled={!playingSync || !paused}
>
  Resume
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

      <div style={{ display: "flex", gap: 24, alignItems: "flex-start", marginTop: 16 }}>
        {/* FSRCNN output */}
        {fsrcnnFrameUrl && (
          <div>
            <h3>FSRCNN Upscale</h3>
            <img
              src={fsrcnnFrameUrl}
              alt="fsrcnn"
              width={frameMeta.w * scaleFactor}
              height={frameMeta.h * scaleFactor}
              style={{
                border: "2px solid #c90",
                background: "#222",
                objectFit: "contain",
              }}
            />
          </div>
        )}
        {/* ESPCN output */}
        {espcnFrameUrl && (
          <div>
            <h3>ESPCN Upscale</h3>
            <img
              src={espcnFrameUrl}
              alt="espcn"
              width={frameMeta.w * scaleFactor}
              height={frameMeta.h * scaleFactor}
              style={{
                border: "2px solid #39c",
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

export default VideoSRCompareSync;
