# MIT License
#
# Copyright (c) 2023 liujing04
# Copyright (c) 2023 源文雨
# Copyright (c) 2023 Ftps
#
# Adapted from:
# https://github.com/RVC-Project/Retrieval-based-Voice-Conversion @ 7b284a6
# Model weights: https://huggingface.co/lj1995/VoiceConversionWebUI

"""RMVPE model and feature extraction (ONNX Runtime backend).

This module implements a pure NumPy/SciPy mel spectrogram frontend and an
ONNX Runtime-based RMVPE pitch estimator.

- ``MelSpectrogram`` computes log-mel spectrograms without PyTorch.
- ``RMVPE`` performs F0 estimation using a pre-trained ONNX model.

References
----------
- RVC Project (MIT License), code:
  https://github.com/RVC-Project/Retrieval-based-Voice-Conversion
  Copyright (c) 2023 liujing04, 源文雨, Ftps

- VoiceConversionWebUI (MIT License), model:
  https://huggingface.co/lj1995/VoiceConversionWebUI
  Copyright (c) 2022 lj1995
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
from librosa.filters import mel as librosa_mel
from scipy.signal import get_window

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError(
        "onnxruntime is not installed. "
        "Please install it through the instructions from https://onnxruntime.ai/, "
        "or simply run `pip install onnxruntime` for CPU-only inference."
    ) from None

_ORT_MIN_VERSION = (1, 17)
_ort_version = tuple(int(x) for x in ort.__version__.split(".")[:2])
if _ort_version < _ORT_MIN_VERSION:
    raise RuntimeError(
        f"onnxruntime >= 1.17 is required, found {ort.__version__}. "
        "Please upgrade onnxruntime."
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MEL_CHANNELS = 128
_SAMPLE_RATE  = 16_000
_WIN_LENGTH   = 1024
_HOP_LENGTH   = 160
_MEL_FMIN     = 30
_MEL_FMAX     = 8_000
_N_CLASS      = 360
_CENTS_OFFSET = 1997.3794084376191

_PROVIDER_MAP = {
    "cuda":       "CUDAExecutionProvider",
    "dml":        "DmlExecutionProvider",
    "rocm":       "ROCMExecutionProvider",
    "coreml":     "CoreMLExecutionProvider",
    "tensorrt":   "TensorrtExecutionProvider",
    "openvino":   "OpenVINOExecutionProvider",
    "cpu":        "CPUExecutionProvider",
}


# ---------------------------------------------------------------------------
# Mel spectrogram (pure NumPy)
# ---------------------------------------------------------------------------

class MelSpectrogram:
    """Pure NumPy/SciPy mel spectrogram — no PyTorch dependency."""

    def __init__(
        self,
        n_mel_channels: int         = _MEL_CHANNELS,
        sampling_rate:  int         = _SAMPLE_RATE,
        win_length:     int         = _WIN_LENGTH,
        hop_length:     int         = _HOP_LENGTH,
        n_fft:          int | None  = None,
        mel_fmin:       float       = _MEL_FMIN,
        mel_fmax:       float       = _MEL_FMAX,
        clamp:          float       = 1e-5,
    ):
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft      = n_fft or win_length
        self.clamp      = clamp

        self.mel_basis = librosa_mel(
            sr=sampling_rate,
            n_fft=self.n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        ).astype(np.float32)  # (n_mels, n_fft//2+1)

    def __call__(
        self,
        audio:    np.ndarray,
        keyshift: float = 0,
        speed:    float = 1,
        center:   bool  = True,
    ) -> np.ndarray:
        """Compute log-mel spectrogram from audio.

        Parameters
        ----------
        audio : np.ndarray [shape=(N,)]
            Mono audio signal.
        keyshift : float, optional
            Pitch shift in semitones. Affects FFT size and frequency scaling.
            Default is 0 (no shift).
        speed : float, optional
            Time-stretch factor. Affects hop length. Default is 1 (no change).
        center : bool, optional
            If True, pad the signal so frames are centered. Default is True.

        Returns
        -------
        np.ndarray [shape=(n_mels, T)]
            Log-mel spectrogram.

        Notes
        -----
        - Uses a Hann window and strided STFT implementation.
        - Output is natural log of mel energies with lower bound clipping.
        - Frequency resolution may change when ``keyshift`` is applied.

        Examples
        --------
        >>> import numpy as np
        >>> from rmvpe_onnx import MelSpectrogram
        >>> sr = 16000
        >>> t = np.linspace(0, 1, sr, endpoint=False)
        >>> audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        >>> mel = MelSpectrogram()
        >>> spec = mel(audio)
        >>> spec.shape[0]
        128
        """
        factor         = 2 ** (keyshift / 12)
        n_fft_new      = int(np.round(self.n_fft      * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))

        window = get_window("hann", win_length_new, fftbins=True).astype(np.float32)
        if n_fft_new > win_length_new:
            pad    = n_fft_new - win_length_new
            window = np.pad(window, (pad // 2, pad - pad // 2))

        if center:
            audio = np.pad(audio, n_fft_new // 2, mode="reflect")

        # Strided STFT
        shape   = ((audio.size - n_fft_new) // hop_length_new + 1, n_fft_new)
        strides = (audio.strides[0] * hop_length_new, audio.strides[0])
        frames  = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
        fft_out   = np.fft.rfft(frames * window, n=n_fft_new)      # (frames, n_fft//2+1)
        magnitude = np.abs(fft_out).T.astype(np.float32)           # (n_fft//2+1, frames)

        if keyshift != 0:
            size = self.n_fft // 2 + 1
            if magnitude.shape[0] < size:
                magnitude = np.pad(magnitude, ((0, size - magnitude.shape[0]), (0, 0)))
            magnitude = magnitude[:size] * self.win_length / win_length_new

        mel_out = self.mel_basis @ magnitude                        # (n_mels, frames)
        return np.log(np.clip(mel_out, self.clamp, None))


# ---------------------------------------------------------------------------
# RMVPE
# ---------------------------------------------------------------------------

class RMVPE:
    """RMVPE pitch estimator using ONNX Runtime.

    This class estimates fundamental frequency (F0) from audio using a
    pre-trained RMVPE ONNX model. It provides a lightweight alternative
    to PyTorch-based implementations and follows a similar interface to
    ``crepe.predict()``.

    Parameters
    ----------
    model_path : str or Path or None, optional
        Path to ``rmvpe.onnx``.

        - ``None``: use the default model path
        - If the file does not exist, it will be downloaded automatically
        - Custom paths and filenames are supported

    device : str or None, optional
        Execution device for ONNX Runtime.

        Supported values include:
        ``'cpu'``, ``'cuda'``, ``'cuda:1'``, ``'dml'``, ``'rocm'``,
        ``'coreml'``, ``'tensorrt'``, ``'openvino'``.

        - ``None``: automatically select the best available provider

    Notes
    -----
    - Requires ``onnxruntime >= 1.17``.
    - Uses a NumPy-based mel spectrogram frontend (no PyTorch dependency).
    - Audio is internally resampled to 16 kHz and downmixed to mono.
    - Frame hop is 160 samples (~10 ms at 16 kHz).
    - The model is loaded via ``ensure_model()`` and cached locally.

    Examples
    --------
    >>> import soundfile as sf
    >>> from rmvpe_onnx import RMVPE
    >>> audio, sr = sf.read("assets/example.wav")  # doctest: +SKIP
    >>> rmvpe = RMVPE()  # doctest: +SKIP
    >>> time, frequency, confidence, activation = rmvpe.predict(audio, sr)  # doctest: +SKIP
    >>> len(time) == len(frequency) == len(confidence)  # doctest: +SKIP
    True
    """

    _SAMPLE_RATE = _SAMPLE_RATE

    def __init__(
        self,
        model_path: str | Path | None = None,
        device:     str | None = None,
    ):
        from .weights import ensure_model
        self.model_path = ensure_model(model_path)

        if device is None:
            device = self._auto_device()

        provider     = self._pick_provider(device)

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            self.model_path, sess_options=sess_opts, providers=[provider]
        )
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.mel_extractor = MelSpectrogram()

        cents              = 20 * np.arange(_N_CLASS) + _CENTS_OFFSET
        self.cents_mapping = np.pad(cents, (4, 4))  # length 368

        logger.info("RMVPE ready — provider: %s", self.session.get_providers()[0])

    # ------------------------------------------------------------------
    # Device helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_device() -> str:
        """Select the best available execution device automatically.

        Iterates ``_PROVIDER_MAP`` in insertion order, which defines the
        priority: ``cuda`` > ``dml`` > ``rocm`` > ``coreml`` > ``tensorrt`` >
        ``openvino`` > ``cpu``.  The first non-CPU provider whose ORT backend
        is available is returned; falls back to ``'cpu'`` if none are found.
        """
        available = ort.get_available_providers()
        for device, provider in _PROVIDER_MAP.items():
            if device == "cpu":
                continue
            if provider in available:
                return device
        return "cpu"

    @staticmethod
    def _pick_provider(device: str):
        available = ort.get_available_providers()
        device_lower = device.lower()

        # strip device index — "cuda:1" → "cuda"
        device_key = device_lower.split(":")[0]

        if device_key not in _PROVIDER_MAP:
            raise ValueError(
                f"Unknown device '{device}'. "
                f"Valid options: {list(_PROVIDER_MAP.keys())}"
            )

        provider_name = _PROVIDER_MAP[device_key]

        if provider_name not in available:
            raise RuntimeError(
                f"{provider_name} is not available in this onnxruntime installation.\n"
                f"Available providers: {available}"
            )

        # provider-specific options
        options = {}
        if device_key == "cuda":
            options["device_id"] = int(device_lower.split(":")[-1]) if ":" in device_lower else 0
        elif device_key == "tensorrt":
            options["device_id"] = int(device_lower.split(":")[-1]) if ":" in device_lower else 0
            options["trt_fp16_enable"] = True
        elif device_key == "openvino":
            options["device_type"] = "GPU" if ":" not in device_lower else device_lower.split(":")[-1].upper()

        return (provider_name, options) if options else provider_name

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _mel2hidden(self, mel: np.ndarray) -> np.ndarray:
        n_frames = mel.shape[-1]
        if n_frames < 32:
            logger.warning(
                "Input audio is very short (%d mel frame(s)). "
                "The output will be heavily padded and may be unreliable. "
                "Consider passing at least %.2f seconds of audio.",
                n_frames,
                32 * _HOP_LENGTH / _SAMPLE_RATE,
            )
        n_pad    = 32 * ((n_frames - 1) // 32 + 1) - n_frames
        if n_pad:
            mel = np.pad(mel, ((0, 0), (0, 0), (0, n_pad)))
        hidden = self.session.run(
            [self.output_name], {self.input_name: mel}
        )[0]
        return hidden[:, :n_frames]

    def _to_local_average_cents(self, salience: np.ndarray) -> np.ndarray:
        center   = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center  += 4

        frame_idx   = np.arange(salience.shape[0])[:, None]
        bin_offsets = np.arange(9)[None, :]
        bin_idx     = (center[:, None] - 4) + bin_offsets

        window_sal  = salience[frame_idx, bin_idx]
        window_cent = self.cents_mapping[bin_idx]

        return (window_sal * window_cent).sum(1) / window_sal.sum(1)

    def predict(
        self,
        audio:  np.ndarray,
        sr:     int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Estimate fundamental frequency (F0) from audio.

        This method follows a similar interface to ``crepe.predict()``,
        returning time, frequency, confidence, and activation.

        Parameters
        ----------
        audio : np.ndarray [shape=(N,) or (N, C)]
            Audio samples. Multichannel audio will be downmixed to mono.
            Expected dtype is float-like; values are typically in ``[-1, 1]``.
        sr : int
            Sample rate of the input audio. Audio will be resampled to
            16 kHz internally if needed.

        Returns
        -------
        time : np.ndarray [shape=(T,)]
            Timestamps in seconds for each frame (~10 ms resolution).
        frequency : np.ndarray [shape=(T,)]
            Estimated pitch in Hz.
        confidence : np.ndarray [shape=(T,)]
            Voicing confidence in the range ``[0, 1]``.
        activation : np.ndarray [shape=(T, 360)]
            Raw salience over pitch bins (~20-cent resolution).

        Notes
        -----
        - Internally resamples audio to 16 kHz.
        - Uses a hop length of 160 samples (~10 ms per frame).
        - Pitch is computed via local averaging in the log-frequency domain.
        - Unvoiced frames may have low confidence and unstable frequency.

        Examples
        --------
        >>> import soundfile as sf
        >>> from rmvpe_onnx import RMVPE
        >>> audio, sr = sf.read("assets/example.wav")  # doctest: +SKIP
        >>> rmvpe = RMVPE()  # doctest: +SKIP
        >>> time, frequency, confidence, activation = rmvpe.predict(audio, sr)  # doctest: +SKIP
        >>> len(time) == len(frequency) == len(confidence)  # doctest: +SKIP
        True
        """
        # downmix + resample
        if audio.ndim > 1:
            audio = librosa.to_mono(audio.T)
        if sr != _SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=_SAMPLE_RATE)
        audio = audio.astype(np.float32)

        mel        = self.mel_extractor(audio)[np.newaxis]        # (1, 128, T) — batch × mel_bins × frames
        # _mel2hidden returns (batch, T, 360); drop batch dim → (T, 360) salience matrix
        activation = self._mel2hidden(mel)[0].astype(np.float32)  # (T, 360)
        confidence = activation.max(axis=1)                       # (T,)

        cents      = self._to_local_average_cents(activation)
        frequency  = 10 * (2 ** (cents / 1200))

        hop_seconds = _HOP_LENGTH / _SAMPLE_RATE
        time        = np.arange(len(frequency)) * hop_seconds     # (T,)

        return time, frequency, confidence, activation
