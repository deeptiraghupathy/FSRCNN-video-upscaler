importScripts('/onnxruntime-web/ort.min.js');
    let session = null, modelLoaded = false, lastShape = null, outputBuf = null;
    onmessage = async function(e) {
      const {type, data} = e.data;
      if(type === 'init') {
        session = await ort.InferenceSession.create(data.modelUrl, {executionProviders: ["wasm"]});
        modelLoaded = true;
        postMessage({type: 'init-done'});
      } else if(type === 'infer') {
        if(!modelLoaded) return postMessage({type:'fail'});
        const {tensorY, u, v, shape, id} = data;
        // Only realloc outputBuf if needed
        if(!lastShape || lastShape[0] !== shape[0] || lastShape[1] !== shape[1]) {
          outputBuf = new Uint8ClampedArray(shape[0]*shape[1]*4);
          lastShape = shape.slice();
        }
        let upY = null, outW=0, outH=0;
        try {
          const outputs = await session.run({ input: new ort.Tensor('float32', tensorY, [1,1,shape[1],shape[0]]) });
          const out = outputs.output || Object.values(outputs)[0];
          upY = out.data; outW = out.dims[3]; outH = out.dims[2];
        } catch { postMessage({type:'fail'}); return; }

        // Upscale U/V (nearest) and merge
        const upU = new Float32Array(outW*outH), upV = new Float32Array(outW*outH);
        for(let y=0;y<outH;++y) for(let x=0;x<outW;++x) {
          const sx = Math.floor(x*shape[0]/outW), sy = Math.floor(y*shape[1]/outH), si = sy*shape[0]+sx, di = y*outW+x;
          upU[di]=u[si]; upV[di]=v[si];
        }
        // CPU fallback:
        for(let i=0;i<outW*outH;++i) {
          const Y = upY[i]*255, U = upU[i]-128, V = upV[i]-128;
          let r = Y+1.402*V, g=Y-0.344136*U-0.714136*V, b=Y+1.772*U;
          outputBuf[i*4]=Math.max(0,Math.min(255,Math.round(r)));
          outputBuf[i*4+1]=Math.max(0,Math.min(255,Math.round(g)));
          outputBuf[i*4+2]=Math.max(0,Math.min(255,Math.round(b)));
          outputBuf[i*4+3]=255;
        }
        postMessage({type:'done', id, buf:outputBuf.buffer, w:outW, h:outH}, [outputBuf.buffer]);
      }
    };