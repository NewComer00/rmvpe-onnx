# rmvpe-onnx

RMVPE pitch estimator — pure ONNX Runtime inference, no PyTorch required.

A simple wrapper around ONNX-related code in [`rvc/lib/rmvpe.py @ 7e03261`](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion/blob/7e03261/rvc/lib/rmvpe.py), [RVC-Project/Retrieval-based-Voice-Conversion](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion)  
Copyright (c) 2023 liujing04, 源文雨, Ftps — MIT License

ONNX model from [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI)  
Copyright (c) 2022 lj1995 — MIT License

---

## Install

To install `rmvpe-onnx` with the [CLI tool](#cli), run:

```bash
pip install rmvpe-onnx[cli]
```

For only the [Python API](#python-api) without CLI, run:

```bash
pip install rmvpe-onnx
```

> [!TIP]
> The installation includes the ONNX Runtime CPU backend `onnxruntime`, which is sufficient for most use cases.
>
> To use hardware acceleration (e.g. CUDA, DirectML, OpenVINO), you must additionally install a compatible ONNX Runtime variant and configure the execution provider accordingly. See the [ONNX Runtime documentation](https://onnxruntime.ai/) for setup instructions.

## CLI

A command-line interface is provided for convenience, which will become usable after installing with the `[cli]` extra.

### Download model

This command downloads `rmvpe.onnx` into the default model path
(`rmvpe_onnx.default_model_path()`) or to a custom path.

Show usage instructions:

```bash
rmvpe-onnx download --help
```

Basic usage (downloads to default model path):

```bash
rmvpe-onnx download
```

Or specify a custom destination path:

```bash
rmvpe-onnx download --model /opt/models/rmvpe.onnx
```

### Pitch estimation

This command runs pitch estimation on a given audio file on the specified device, with optional confidence thresholding, plotting, and CSV output.

Show usage instructions:

```bash
rmvpe-onnx predict --help
```

Basic usage (default model path, auto device, default thresholding, no plot, no CSV):

```bash
rmvpe-onnx predict assets/example.wav
```

Specify a custom model path and use CPU device for inference:

```bash
rmvpe-onnx predict assets/example.wav --model /opt/models/rmvpe.onnx --device cpu
```

Run pitch estimation on first CUDA device (`onnxruntime-gpu` required) and show an interactive plot:

```bash
rmvpe-onnx predict assets/example.wav --device cuda --plot
```

Run pitch estimation with confidence thresholding and save results (time, frequency, confidence) as CSV. The frequency values will be set to 0 where confidence is below the threshold:

```bash
rmvpe-onnx predict assets/example.wav --confidence_threshold 0.05 --csv output.csv
```

> [!TIP]
> The ONNX model will be automatically downloaded if the model file is not found in the default model path or the specified `--model` path.

## Python API

`rmvpe_onnx` also provides a Python API for pitch estimation, much alike to [marl/CREPE](https://github.com/marl/crepe) but simpler.

### Usage

```python
from rmvpe_onnx import RMVPE
import soundfile as sf

# Load the example audio file (replace with your own path)
audio, sr = sf.read("assets/example.wav")

# Initialize the RMVPE model
rmvpe = RMVPE(model_path=None, device=None)

# Specify model_path and device if needed, e.g.:
# rmvpe = RMVPE(model_path="/opt/models/rmvpe.onnx", device="cpu")

# Run pitch estimation
time, frequency, confidence, activation = rmvpe.predict(audio=audio, sr=sr)
```

### `RMVPE(model_path=None, device=None)`

| Parameter    | Type                  | Default | Description                                                                                                                                                                                   |
|--------------|-----------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model_path` | `str \| Path \| None` | `None`  | Path to `rmvpe.onnx`. If `None`, uses the default model path. If the model does not exist at the given path, it will be downloaded and saved there automatically during class initialization. |
| `device`     | `str \| None`         | `None`  | `'cpu'`, `'cuda'`, `'cuda:1'`, `'dml'`, `'rocm'`, `'coreml'`, `'tensorrt'`, `'openvino'`. Auto-detected if `None`.                                                                            |

### `RMVPE.predict(audio, sr)`

| Parameter | Type         | Default | Description                                                         |
|-----------|--------------|---------|---------------------------------------------------------------------|
| `audio`   | `np.ndarray` | —       | Audio samples, shape `(N,)` or `(N, C)`. Multichannel is downmixed. |
| `sr`      | `int`        | —       | Sample rate. Resampled to 16 kHz internally if needed.              |

Returns a 4-tuple:

| Name         | Shape      | Description                                 |
|--------------|------------|---------------------------------------------|
| `time`       | `(T,)`     | Timestamps in seconds, spaced by 0.01 s.    |
| `frequency`  | `(T,)`     | Pitch in Hz.                                |
| `confidence` | `(T,)`     | Voicing confidence in `[0, 1]`.             |
| `activation` | `(T, 360)` | Raw salience matrix over 360 pitch classes. |
