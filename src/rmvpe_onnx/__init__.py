"""RMVPE pitch estimator with pure ONNX Runtime inference.

This package provides a lightweight wrapper around the ONNX-based RMVPE
implementation from the RVC project, along with a pre-converted ONNX model.

References
----------
- RVC Project (MIT License), code:
  https://github.com/RVC-Project/Retrieval-based-Voice-Conversion
  Copyright (c) 2023 liujing04, 源文雨, Ftps

- VoiceConversionWebUI (MIT License), model:
  https://huggingface.co/lj1995/VoiceConversionWebUI
  Copyright (c) 2022 lj1995
"""

from .model import RMVPE, MelSpectrogram
from .weights import default_model_path, ensure_model

__version__ = "0.2.3"

__all__ = [
    "RMVPE",
    "MelSpectrogram",
    "ensure_model",
    "default_model_path",
]
