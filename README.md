# rmvpe-onnx

RMVPE pitch estimator — pure ONNX Runtime inference, no PyTorch required.

A simple wrapper around ONNX-related code in [`rvc/lib/rmvpe.py @ 7e03261`](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion/blob/7e03261/rvc/lib/rmvpe.py), [RVC-Project/Retrieval-based-Voice-Conversion](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion)  
Copyright (c) 2023 liujing04, 源文雨, Ftps — MIT License

ONNX model from [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI)  
Copyright (c) 2022 lj1995 — MIT License

---

## Install

```bash
pip install rmvpe-onnx        # Python API only
pip install rmvpe-onnx[cli]   # with CLI tool
```

> [!TIP]
> Includes `onnxruntime` (CPU). For hardware acceleration (CUDA, DirectML, etc.), install a compatible ONNX Runtime variant. See the [ONNX Runtime documentation](https://onnxruntime.ai/).

## CLI

```bash
# Download the ONNX model
# [optional] auto-downloaded on first predict if skipped
rmvpe-onnx download

# Run pitch prediction with default settings and plot the results
rmvpe-onnx predict audio.wav --plot
```

For all options, see the [CLI Reference](https://NewComer00.github.io/rmvpe-onnx/cli.html) or run `rmvpe-onnx download --help` and `rmvpe-onnx predict --help`.

## Python API

> [!NOTE]
> The Python API returns raw outputs with no confidence thresholding applied. Use `confidence` to filter `frequency` yourself if needed.

```python
from rmvpe_onnx import RMVPE
import soundfile as sf

audio, sr = sf.read("audio.wav")
rmvpe = RMVPE()

time, frequency, confidence, activation = rmvpe.predict(audio=audio, sr=sr)

# Optional: zero out frequency where confidence is below a threshold
# frequency[confidence < 0.03] = 0.0
```

For full parameter reference and return values, see the [API Reference](https://NewComer00.github.io/rmvpe-onnx/api.html).
