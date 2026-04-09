"""Model weight management utilities for RMVPE.

This module provides helper functions to locate and download the RMVPE
ONNX model used for inference.

- ``default_model_path()`` returns the default location inside the package.
- ``ensure_model()`` ensures the model exists locally, downloading it if needed.

Notes
-----
- The model is downloaded from Hugging Face:
  https://huggingface.co/lj1995/VoiceConversionWebUI
- Downloads occur only if the target file does not already exist.
- For reproducibility, consider pinning a specific model revision or
  verifying checksums in downstream applications.

References
----------
- VoiceConversionWebUI (MIT License), model:
  https://huggingface.co/lj1995/VoiceConversionWebUI
  Copyright (c) 2022 lj1995
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_HF_REPO     = "lj1995/VoiceConversionWebUI"
_HF_FILENAME = "rmvpe.onnx"


def default_model_path() -> Path:
    """Return the default path to ``rmvpe.onnx``.

    Returns
    -------
    Path
        Absolute path to the model file inside the package data directory.

    Notes
    -----
    This function does not check whether the file exists.
    Use ``ensure_model()`` to guarantee availability.
    """
    return Path(__file__).parent / "data" / _HF_FILENAME


def ensure_model(model_path: str | Path | None = None) -> str:
    """Ensure the RMVPE ONNX model exists locally.

    If the model file does not exist at the specified location, it will be
    downloaded from Hugging Face and saved to that path.

    Parameters
    ----------
    model_path : str or Path or None, optional
        Target path for the model file.

        - ``None``: use the default path from ``default_model_path()``
        - If the file does not exist, it will be downloaded
        - Custom filenames are supported

    Returns
    -------
    str
        Absolute path to the model file.

    Notes
    -----
    - Parent directories are created automatically
    - If the file already exists, no download is performed
    - No validation is done on the file contents; if the file exists, it is assumed to be correct

    Examples
    --------
    Use the default location:

    >>> from rmvpe_onnx import ensure_model
    >>> path = ensure_model()  # doctest: +SKIP

    Use a custom path:

    >>> path = ensure_model("/tmp/rmvpe.onnx")  # doctest: +SKIP

    Use a custom filename:

    >>> path = ensure_model("/opt/models/pitch_detector.onnx")  # doctest: +SKIP
    """
    from huggingface_hub import hf_hub_download

    dest = Path(model_path) if model_path is not None else default_model_path()

    if dest.exists():
        logger.debug("Model found: %s", dest)
        return str(dest)

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s from %s ...", _HF_FILENAME, _HF_REPO)
    hf_path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_FILENAME)
    shutil.copy(hf_path, dest)
    logger.info("Saved → %s", dest)
    return str(dest)
