import React, { useState, useRef } from 'react';
import * as ort from 'onnxruntime-web';

const FSRCNNTest = () => {
  const [originalImage, setOriginalImage] = useState(null);
  const [upscaledImage, setUpscaledImage] = useState(null);
  const [classicUpscaledImage, setClassicUpscaledImage] = useState(null);
  const [neuralColorUpscaledImage, setNeuralColorUpscaledImage] = useState(null);
  const [scaleFactor, setScaleFactor] = useState(4); //Change for different scale
  const [isProcessing, setIsProcessing] = useState(false);
  const fileInputRef = useRef(null);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        setOriginalImage(event.target.result);
        setUpscaledImage(null);
        setClassicUpscaledImage(null);
        setNeuralColorUpscaledImage(null);
      };
      reader.readAsDataURL(file);
    }
  };

  // === YUV <-> RGB Utilities ===
  function rgbToYuv(r, g, b) {
    // BT.601 conversion
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    const u = -0.168736 * r - 0.331264 * g + 0.5 * b + 128;
    const v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128;
    return [y, u, v];
  }

  function yuvToRgb(y, u, v) {
    // Inverse BT.601
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

  // === Classic Upscale (Canvas) ===
  const classicUpscale = (imgUrl, factor) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width * factor;
        canvas.height = img.height * factor;
        const ctx = canvas.getContext('2d');
        ctx.imageSmoothingEnabled = true;
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL());
      };
      img.onerror = reject;
      img.src = imgUrl;
    });
  };

  // === Utility to Upscale 2D Channel Data ===
  // const upscaleChannel = (channelData, width, height, scale) => {
  //   // channelData: Float32Array of shape width*height
  //   // Returns: Float32Array of shape (width*scale)*(height*scale)
  //   const srcCanvas = document.createElement('canvas');
  //   srcCanvas.width = width;
  //   srcCanvas.height = height;
  //   const srcCtx = srcCanvas.getContext('2d');
  //   const imgData = srcCtx.createImageData(width, height);
  //   for (let i = 0; i < width * height; i++) {
  //     const v = Math.round(channelData[i]);
  //     imgData.data[i * 4 + 0] = v;
  //     imgData.data[i * 4 + 1] = v;
  //     imgData.data[i * 4 + 2] = v;
  //     imgData.data[i * 4 + 3] = 255;
  //   }
  //   srcCtx.putImageData(imgData, 0, 0);
  
  //   const dstCanvas = document.createElement('canvas');
  //   dstCanvas.width = width * scale;
  //   dstCanvas.height = height * scale;
  //   const dstCtx = dstCanvas.getContext('2d');
  //   dstCtx.imageSmoothingEnabled = false;  // **Force nearest neighbor!**
  //   dstCtx.drawImage(srcCanvas, 0, 0, dstCanvas.width, dstCanvas.height);
  
  //   const upImgData = dstCtx.getImageData(0, 0, dstCanvas.width, dstCanvas.height);
  //   const upArr = new Float32Array(dstCanvas.width * dstCanvas.height);
  //   for (let i = 0; i < upArr.length; i++) {
  //     upArr[i] = upImgData.data[i * 4]; // only need R channel
  //   }
  //   return upArr;
  // };  

  const upscaleChannelToSize = (channelData, srcW, srcH, dstW, dstH) => {
    // channelData: Float32Array, srcW x srcH
    // returns Float32Array, dstW x dstH using nearest neighbor
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
  
  // === Main Upscale Action ===
  const upscaleImage = async () => {
    if (!originalImage) return;
    setIsProcessing(true);

    try {
      // 1. Classic upscale (for reference)
      const classicUpscaledUrl = await classicUpscale(originalImage, scaleFactor);
      setClassicUpscaledImage(classicUpscaledUrl);

      // 2. Load + YUV split
      const { tensorY, uPlane, vPlane, width, height } = await loadAndPreprocessImageAndUV(originalImage);

      // 3. Neural upscaling on Y
      ort.env.wasm.numThreads = 1;
      const session = await ort.InferenceSession.create('/models/fsrcnn_x4.onnx'); //Change for different scale
      const outputs = await session.run({ input: tensorY });
      const outputTensor = outputs.output || Object.values(outputs)[0];

      // Y output: [1,1,H,W], normalized [0,1]
      const outW = outputTensor.dims[3];
      const outH = outputTensor.dims[2];
      const upY = outputTensor.data;

      // 4. Upscale U and V using canvas (classic)
      const upU = upscaleChannelToSize(uPlane, width, height, outW, outH);
      const upV = upscaleChannelToSize(vPlane, width, height, outW, outH);      

      // 5. Merge upY (float, [0,1]) + upU, upV (0-255) to RGB
      const colorUpscaleUrl = yuvMergeAndToDataURL(upY, upU, upV, outW, outH);
      setNeuralColorUpscaledImage(colorUpscaleUrl);

      // 6. Show also neural Y-only grayscale output (optional)
      const upscaledImageUrl = await postProcessOutput(outputTensor, outW, outH);
      setUpscaledImage(upscaledImageUrl);

    } catch (error) {
      console.error('Detailed error:', error);
      alert(`Failed to upscale: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  // === Preprocess: Get Y for NN, U+V for classic ===
  const loadAndPreprocessImageAndUV = (imageUrl) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, img.width, img.height);
        const { data, width, height } = imageData;

        const yArr = new Float32Array(width * height);
        const uArr = new Float32Array(width * height);
        const vArr = new Float32Array(width * height);

        for (let i = 0; i < width * height; i++) {
          const r = data[i * 4 + 0];
          const g = data[i * 4 + 1];
          const b = data[i * 4 + 2];
          const [y, u, v] = rgbToYuv(r, g, b);
          yArr[i] = y / 255.0; // NN expects [0,1]
          uArr[i] = u;         // for upscaling, keep 0-255
          vArr[i] = v;
        }
        // Neural net input tensor
        const tensorY = new ort.Tensor('float32', yArr, [1, 1, height, width]);
        resolve({ tensorY, uPlane: uArr, vPlane: vArr, width, height });
      };
      img.onerror = () => reject("Failed to load image");
      img.src = imageUrl;
    });
  };

  // === Merge upscaled Y, U, V to RGB and return DataURL ===
  const yuvMergeAndToDataURL = (upY, upU, upV, width, height) => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(width, height);
  
    for (let i = 0; i < width * height; i++) {
      const y = upY[i] * 255.0; // Denormalize
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
  

  // === Just NN Y to Grayscale for reference ===
  const postProcessOutput = (outputTensor, width, height) => {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(width, height);

    const tensorData = outputTensor.data; // Float32Array

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = (y * width + x) * 4;
        const val = Math.min(255, Math.max(0, tensorData[y * width + x] * 255));
        imageData.data[idx] = val;
        imageData.data[idx + 1] = val;
        imageData.data[idx + 2] = val;
        imageData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL();
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>FSRCNN Color Upscaling Test</h1>

      <div>
        <input
          type="file"
          accept="image/*"
          ref={fileInputRef}
          onChange={handleImageUpload}
          style={{ display: 'none' }}
        />
        <button onClick={() => fileInputRef.current.click()}>
          Select Image
        </button>
      </div>

      <div style={{ margin: '10px 0' }}>
        <label>Scale Factor: </label>
        <select
          value={scaleFactor}
          onChange={(e) => setScaleFactor(Number(e.target.value))}
        >
          <option value="2">2x</option>
          <option value="3">3x</option>
          <option value="4">4x</option>
        </select>
      </div>

      <button
        onClick={upscaleImage}
        disabled={!originalImage || isProcessing}
      >
        {isProcessing ? 'Processing...' : 'Upscale Image'}
      </button>

      <div style={{ display: 'flex', marginTop: '20px' }}>
        {originalImage && (
          <div style={{ marginRight: '20px', textAlign: 'center' }}>
            <h3>Original</h3>
            <img
              src={originalImage}
              alt="Original"
              style={{ maxWidth: '300px', maxHeight: '300px' }}
            />
          </div>
        )}

        {classicUpscaledImage && (
          <div style={{ marginRight: '20px', textAlign: 'center' }}>
            <h3>Classic Upscale ({scaleFactor}x)</h3>
            <img
              src={classicUpscaledImage}
              alt="Classic Upscaled"
              style={{
                maxWidth: `${300 * scaleFactor}px`,
                maxHeight: `${300 * scaleFactor}px`
              }}
            />
          </div>
        )}

        {upscaledImage && (
          <div style={{ marginRight: '20px', textAlign: 'center' }}>
            <h3>Neural (Y only, grayscale)</h3>
            <img
              src={upscaledImage}
              alt="Neural Upscaled (Y only)"
              style={{
                maxWidth: `${300 * scaleFactor}px`,
                maxHeight: `${300 * scaleFactor}px`
              }}
            />
          </div>
        )}

        {neuralColorUpscaledImage && (
          <div style={{ textAlign: 'center' }}>
            <h3>Neural (Y + UV, color)</h3>
            <img
              src={neuralColorUpscaledImage}
              alt="Neural Color Upscaled"
              style={{
                maxWidth: `${300 * scaleFactor}px`,
                maxHeight: `${300 * scaleFactor}px`
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default FSRCNNTest;
