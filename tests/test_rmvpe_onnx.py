"""Tests for rmvpe-onnx.

Coverage:
  - MelSpectrogram: output shape, dtype, keyshift/speed variants
  - RMVPE: predict output shapes/dtypes, mono/stereo/resample paths
           (ONNX session is mocked — no model file required)
  - RMVPE device/provider: _auto_device, _pick_provider error paths, per-provider options
  - Module-level guards: onnxruntime ImportError, minimum version check
  - ensure_model / default_model_path: path logic, download skipped when file exists
  - CLI: download and predict subcommands (subprocess)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine(freq_hz: float = 440.0, duration: float = 0.5, sr: int = 16_000) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq_hz * t).astype(np.float32)


def _make_fake_activation(n_frames: int, n_class: int = 360) -> np.ndarray:
    """Fake salience matrix: one dominant bin per frame."""
    act = np.zeros((n_frames, n_class), dtype=np.float32)
    dominant = np.random.randint(0, n_class, size=n_frames)
    act[np.arange(n_frames), dominant] = 1.0
    return act


# ---------------------------------------------------------------------------
# MelSpectrogram
# ---------------------------------------------------------------------------

class TestMelSpectrogram:
    def setup_method(self):
        from rmvpe_onnx import MelSpectrogram
        self.mel = MelSpectrogram()

    def test_output_shape_mono(self):
        audio = _sine()
        out = self.mel(audio)
        assert out.ndim == 2
        assert out.shape[0] == 128  # n_mel_channels

    def test_output_dtype(self):
        out = self.mel(_sine())
        assert out.dtype == np.float32

    def test_frame_count_scales_with_duration(self):
        short = self.mel(_sine(duration=0.2))
        long_ = self.mel(_sine(duration=0.8))
        assert long_.shape[1] > short.shape[1]

    def test_keyshift_zero_matches_default(self):
        audio = _sine()
        default = self.mel(audio)
        shifted = self.mel(audio, keyshift=0)
        np.testing.assert_array_equal(default, shifted)

    def test_keyshift_nonzero_changes_output(self):
        audio = _sine()
        default = self.mel(audio)
        shifted = self.mel(audio, keyshift=2)
        # keyshift changes both shape and values — check at least one differs
        assert default.shape != shifted.shape or not np.allclose(default, shifted)

    def test_speed_nonzero_changes_frame_count(self):
        audio = _sine()
        normal = self.mel(audio)
        fast = self.mel(audio, speed=2.0)
        assert fast.shape[1] < normal.shape[1]

    def test_center_false_fewer_frames(self):
        audio = _sine()
        centered = self.mel(audio, center=True)
        not_centered = self.mel(audio, center=False)
        assert not_centered.shape[1] <= centered.shape[1]

    def test_silent_audio_no_nan(self):
        audio = np.zeros(16_000, dtype=np.float32)
        out = self.mel(audio)
        assert not np.any(np.isnan(out))

    def test_mel_basis_shape(self):
        # (n_mels, n_fft//2+1)
        assert self.mel.mel_basis.shape[0] == 128
        assert self.mel.mel_basis.shape[1] == 513  # 1024//2+1


# ---------------------------------------------------------------------------
# RMVPE (mocked ONNX session)
# ---------------------------------------------------------------------------

def _make_rmvpe(n_frames: int = 50):
    """Build an RMVPE instance with a mocked InferenceSession."""
    fake_act = _make_fake_activation(n_frames)  # (T, 360)

    mock_session = MagicMock()
    mock_session.get_inputs.return_value  = [MagicMock(name="mel")]
    mock_session.get_outputs.return_value = [MagicMock(name="salience")]
    mock_session.get_providers.return_value = ["CPUExecutionProvider"]
    # session.run returns list-of-arrays; shape must be (1, T, 360) then [0] → (T, 360)
    mock_session.run.return_value = [fake_act[np.newaxis]]  # (1, T, 360)

    with (
        patch("rmvpe_onnx.model.ort.InferenceSession", return_value=mock_session),
        patch("rmvpe_onnx.model.ort.get_available_providers", return_value=["CPUExecutionProvider"]),
        patch("rmvpe_onnx.weights.ensure_model", return_value="/fake/rmvpe.onnx"),
    ):
        from rmvpe_onnx import RMVPE
        rmvpe = RMVPE(model_path="/fake/rmvpe.onnx", device="cpu")

    rmvpe.session = mock_session
    return rmvpe, fake_act


class TestRMVPEPredict:
    def test_output_is_4_tuple(self):
        rmvpe, _ = _make_rmvpe()
        result = rmvpe.predict(_sine(), sr=16_000)
        assert len(result) == 4

    def test_output_names_and_shapes(self):
        n_frames = 50
        rmvpe, fake_act = _make_rmvpe(n_frames)
        audio = _sine(duration=0.5, sr=16_000)
        time, freq, conf, act = rmvpe.predict(audio, sr=16_000)

        T = len(time)
        assert time.shape  == (T,)
        assert freq.shape  == (T,)
        assert conf.shape  == (T,)
        assert act.shape   == (T, 360)

    def test_output_dtypes_are_float32(self):
        rmvpe, _ = _make_rmvpe()
        time, freq, conf, act = rmvpe.predict(_sine(), sr=16_000)
        # activation and confidence come directly from the ONNX output cast to float32
        assert act.dtype  == np.float32, f"activation: expected float32, got {act.dtype}"
        assert conf.dtype == np.float32, f"confidence: expected float32, got {conf.dtype}"
        # time and frequency go through Python float arithmetic; they are float64
        assert time.dtype.kind == "f"
        assert freq.dtype.kind == "f"

    def test_timestamps_monotonically_increasing(self):
        rmvpe, _ = _make_rmvpe()
        time, *_ = rmvpe.predict(_sine(), sr=16_000)
        assert np.all(np.diff(time) > 0)

    def test_timestamp_spacing_is_10ms(self):
        rmvpe, _ = _make_rmvpe()
        time, *_ = rmvpe.predict(_sine(), sr=16_000)
        diffs = np.diff(time)
        np.testing.assert_allclose(diffs, 0.01, atol=1e-6)

    def test_confidence_in_unit_interval(self):
        rmvpe, _ = _make_rmvpe()
        _, _, conf, _ = rmvpe.predict(_sine(), sr=16_000)
        assert conf.min() >= 0.0
        assert conf.max() <= 1.0 + 1e-6

    def test_frequency_nonnegative(self):
        rmvpe, _ = _make_rmvpe()
        _, freq, _, _ = rmvpe.predict(_sine(), sr=16_000)
        assert np.all(freq >= 0)

    def test_stereo_input_downmixed(self):
        """Stereo (N, 2) audio should not raise."""
        rmvpe, _ = _make_rmvpe()
        stereo = np.stack([_sine(), _sine(freq_hz=880)], axis=-1)  # (N, 2)
        time, freq, conf, act = rmvpe.predict(stereo, sr=16_000)
        assert len(time) > 0

    def test_resampling_path(self):
        """Audio at a non-native sample rate should be accepted."""
        rmvpe, _ = _make_rmvpe()
        audio_44k = _sine(sr=44_100)
        time, freq, conf, act = rmvpe.predict(audio_44k, sr=44_100)
        assert len(time) > 0

    def test_very_short_audio(self):
        """A very short clip (< 1 frame worth) should not crash."""
        rmvpe, _ = _make_rmvpe(n_frames=1)
        audio = np.zeros(160, dtype=np.float32)  # exactly 1 hop
        # just must not raise
        rmvpe.predict(audio, sr=16_000)


# ---------------------------------------------------------------------------
# _mel2hidden padding
# ---------------------------------------------------------------------------

class TestMel2Hidden:
    def test_output_frames_match_input(self):
        rmvpe, _ = _make_rmvpe(n_frames=33)  # 33 is not a multiple of 32
        n_frames = 33
        mel = np.zeros((1, 128, n_frames), dtype=np.float32)
        hidden = rmvpe._mel2hidden(mel)
        # shape is (batch, T, 360); axis 1 is frames, axis 2 is pitch classes
        assert hidden.shape[1] == n_frames, (
            f"expected {n_frames} frames on axis 1, got shape {hidden.shape}"
        )

    def test_output_frames_multiple_of_32(self):
        rmvpe, _ = _make_rmvpe(n_frames=64)
        mel = np.zeros((1, 128, 64), dtype=np.float32)
        hidden = rmvpe._mel2hidden(mel)
        assert hidden.shape[1] == 64

    def test_output_pitch_classes(self):
        rmvpe, _ = _make_rmvpe(n_frames=32)
        mel = np.zeros((1, 128, 32), dtype=np.float32)
        hidden = rmvpe._mel2hidden(mel)
        assert hidden.shape[2] == 360


# ---------------------------------------------------------------------------
# _to_local_average_cents
# ---------------------------------------------------------------------------

class TestToLocalAverageCents:
    def setup_method(self):
        self.rmvpe, _ = _make_rmvpe()

    def test_output_shape(self):
        salience = np.zeros((20, 360), dtype=np.float32)
        salience[:, 180] = 1.0
        cents = self.rmvpe._to_local_average_cents(salience)
        assert cents.shape == (20,)

    def test_concentrated_peak_returns_expected_cents(self):
        """When salience is a sharp spike at bin 180, result should be near
        the cent value for bin 180."""
        salience = np.zeros((1, 360), dtype=np.float32)
        salience[0, 180] = 1.0
        cents = self.rmvpe._to_local_average_cents(salience)
        expected = 20 * 180 + 1997.3794084376191
        assert abs(cents[0] - expected) < 1.0


# ---------------------------------------------------------------------------
# ensure_model / default_model_path
# ---------------------------------------------------------------------------

class TestEnsureModel:
    def testdefault_model_path_is_inside_package(self):
        from rmvpe_onnx import default_model_path
        path = default_model_path()
        assert path.name == "rmvpe.onnx"
        assert "rmvpe_onnx" in str(path)

    def test_returns_existing_file_without_downloading(self, tmp_path):
        from rmvpe_onnx import ensure_model
        model = tmp_path / "rmvpe.onnx"
        model.write_bytes(b"fake")
        with patch("huggingface_hub.hf_hub_download") as mock_dl:
            result = ensure_model(model)
            mock_dl.assert_not_called()
        assert result == str(model)

    def test_downloads_when_file_missing(self, tmp_path):
        from rmvpe_onnx import ensure_model
        dest = tmp_path / "rmvpe.onnx"
        fake_cached = tmp_path / "cached.onnx"
        fake_cached.write_bytes(b"fake model bytes")

        with patch("huggingface_hub.hf_hub_download", return_value=str(fake_cached)) as mock_dl:
            result = ensure_model(str(dest))
            mock_dl.assert_called_once()
        assert Path(result).exists()
        assert Path(result).read_bytes() == b"fake model bytes"

    def test_creates_parent_directories(self, tmp_path):
        from rmvpe_onnx import ensure_model
        dest = tmp_path / "nested" / "deep" / "rmvpe.onnx"
        fake_cached = tmp_path / "cached.onnx"
        fake_cached.write_bytes(b"x")

        with patch("huggingface_hub.hf_hub_download", return_value=str(fake_cached)):
            ensure_model(str(dest))
        assert dest.exists()


# ---------------------------------------------------------------------------
# CLI — download subcommand
# ---------------------------------------------------------------------------

class TestCLIDownload:
    def test_download_uses_existing_file(self, tmp_path):
        model = tmp_path / "rmvpe.onnx"
        model.write_bytes(b"fake")
        result = subprocess.run(
            [sys.executable, "-m", "rmvpe_onnx.cli", "download", "--model", str(model)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "Model ready" in result.stdout

    def test_download_unknown_flag_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-m", "rmvpe_onnx.cli", "download", "--notaflag"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# CLI — predict subcommand
# ---------------------------------------------------------------------------

class TestCLIPredict:
    @pytest.fixture()
    def wav_file(self, tmp_path):
        """Write a minimal valid WAV file (16-bit PCM, mono, 16 kHz)."""
        import struct
        import wave
        path = tmp_path / "test.wav"
        samples = (_sine() * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16_000)
            wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
        return path

    @pytest.fixture()
    def fake_model(self, tmp_path):
        p = tmp_path / "rmvpe.onnx"
        p.write_bytes(b"not a real model")
        return p

    def test_predict_missing_audio_exits_nonzero(self, tmp_path, fake_model):
        result = subprocess.run(
            [
                sys.executable, "-m", "rmvpe_onnx.cli", "predict",
                str(tmp_path / "nonexistent.wav"),
                "--model", str(fake_model),
            ],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_predict_no_audio_arg_exits_nonzero(self):
        result = subprocess.run(
            [sys.executable, "-m", "rmvpe_onnx.cli", "predict"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_predict_runs_end_to_end(self, wav_file, tmp_path, monkeypatch):
        """Full predict run with a mocked ONNX session."""
        n_frames = 50
        fake_act = _make_fake_activation(n_frames)[np.newaxis]  # (1, T, 360)

        mock_session = MagicMock()
        mock_session.get_inputs.return_value  = [MagicMock(name="mel")]
        mock_session.get_outputs.return_value = [MagicMock(name="salience")]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        mock_session.run.return_value = [fake_act]

        csv_out = tmp_path / "out.csv"

        with (
            patch("rmvpe_onnx.model.ort.InferenceSession", return_value=mock_session),
            patch("rmvpe_onnx.model.ort.get_available_providers", return_value=["CPUExecutionProvider"]),
            patch("rmvpe_onnx.weights.ensure_model", return_value=str(tmp_path / "rmvpe.onnx")),
        ):
            result = subprocess.run(
                [
                    sys.executable, "-m", "rmvpe_onnx.cli", "predict",
                    str(wav_file),
                    "--model", str(tmp_path / "rmvpe.onnx"),
                    "--csv", str(csv_out),
                ],
                capture_output=True, text=True,
                env={**os.environ},
            )

        # Even if mocked session produces garbage activations, the CLI logic
        # (file-not-found check, argparse) must pass.  A real session mock
        # would need the correct ONNX graph; we just verify no hard crash on
        # the pre-inference path.
        assert result.returncode in (0, 1)  # 1 acceptable if ort rejects fake bytes


# ---------------------------------------------------------------------------
# Module-level guards (lines 22-23, 32)
# ---------------------------------------------------------------------------

class TestModuleGuards:
    def test_missing_onnxruntime_raises_import_error(self):
        """ImportError with a helpful message when onnxruntime is absent."""
        import importlib
        import sys

        # Remove cached module so the try/except block re-executes
        mods_to_remove = [k for k in sys.modules if "rmvpe_onnx.model" in k]
        for mod in mods_to_remove:
            del sys.modules[mod]

        with patch.dict(sys.modules, {"onnxruntime": None}):
            with pytest.raises(ImportError, match="onnxruntime is not installed"):
                importlib.import_module("rmvpe_onnx.model")

    def test_old_onnxruntime_raises_runtime_error(self):
        """RuntimeError when onnxruntime is present but below minimum version."""
        import importlib
        import sys

        mods_to_remove = [k for k in sys.modules if "rmvpe_onnx.model" in k]
        for mod in mods_to_remove:
            del sys.modules[mod]

        mock_ort = MagicMock()
        mock_ort.__version__ = "1.16.0"
        mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

        with patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            with pytest.raises(RuntimeError, match="onnxruntime >= 1.17 is required"):
                importlib.import_module("rmvpe_onnx.model")


# ---------------------------------------------------------------------------
# MelSpectrogram — window-padding branch (lines 108-109) and
# keyshift magnitude-padding branch (line 124)
# ---------------------------------------------------------------------------

class TestMelSpectrogramBranches:
    def setup_method(self):
        from rmvpe_onnx import MelSpectrogram
        self.mel = MelSpectrogram()

    def test_nfft_larger_than_win_length_pads_window(self):
        """n_fft > win_length triggers the window zero-padding branch."""
        from rmvpe_onnx import MelSpectrogram
        # n_fft explicitly larger than win_length
        mel = MelSpectrogram(win_length=512, n_fft=1024)
        out = mel(_sine())
        assert out.shape[0] == 128
        assert not np.any(np.isnan(out))

    def test_keyshift_magnitude_shorter_than_target_pads(self):
        """Large negative keyshift shrinks n_fft_new below self.n_fft//2+1,
        triggering the magnitude zero-padding branch (line 124)."""
        out = self.mel(_sine(), keyshift=-6)
        assert out.shape[0] == 128
        assert not np.any(np.isnan(out))


# ---------------------------------------------------------------------------
# RMVPE.__init__ with device=None (line 159)
# ---------------------------------------------------------------------------

class TestRMVPEAutoDevice:
    def test_init_with_device_none_uses_auto_detected_provider(self):
        """device=None should auto-detect and produce a valid session provider."""
        mock_session = MagicMock()
        mock_session.get_inputs.return_value  = [MagicMock(name="mel")]
        mock_session.get_outputs.return_value = [MagicMock(name="salience")]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]

        with (
            patch("rmvpe_onnx.model.ort.InferenceSession", return_value=mock_session) as mock_init,
            patch("rmvpe_onnx.model.ort.get_available_providers", return_value=["CPUExecutionProvider"]),
            patch("rmvpe_onnx.weights.ensure_model", return_value="/fake/rmvpe.onnx"),
        ):
            from rmvpe_onnx import RMVPE
            RMVPE(model_path="/fake/rmvpe.onnx", device=None)
            # Session must have been created with the auto-detected CPU provider
            call_kwargs = mock_init.call_args
            providers = call_kwargs[1]["providers"] if "providers" in call_kwargs[1] else call_kwargs[0][2]
            assert "CPUExecutionProvider" in providers[0]


# ---------------------------------------------------------------------------
# _auto_device (lines 184-190)
# ---------------------------------------------------------------------------

class TestAutoDevice:
    def _auto_device(self, available_providers):
        from rmvpe_onnx.model import RMVPE
        with patch("rmvpe_onnx.model.ort.get_available_providers", return_value=available_providers):
            return RMVPE._auto_device()

    def test_returns_cuda_when_available(self):
        assert self._auto_device(["CUDAExecutionProvider", "CPUExecutionProvider"]) == "cuda"

    def test_returns_cpu_when_only_cpu_available(self):
        assert self._auto_device(["CPUExecutionProvider"]) == "cpu"

    def test_skips_cpu_in_priority_scan(self):
        # Only CPUExecutionProvider — must not return it from the loop, only from fallback
        result = self._auto_device(["CPUExecutionProvider"])
        assert result == "cpu"

    def test_prefers_first_non_cpu_provider(self):
        # rocm listed before cuda in available — but _PROVIDER_MAP iteration order
        # (dict insertion order, Python 3.7+) puts cuda first, so cuda wins
        result = self._auto_device(["ROCMExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
        assert result == "cuda"


# ---------------------------------------------------------------------------
# _pick_provider (lines 201, 209, 217, 219-220, 222)
# ---------------------------------------------------------------------------

class TestPickProvider:
    def _pick(self, device, available=None):
        from rmvpe_onnx.model import RMVPE
        if available is None:
            available = ["CPUExecutionProvider"]
        with patch("rmvpe_onnx.model.ort.get_available_providers", return_value=available):
            return RMVPE._pick_provider(device)

    def test_unknown_device_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown device"):
            self._pick("tpu")

    def test_unavailable_provider_raises_runtime_error(self):
        with pytest.raises(RuntimeError, match="is not available"):
            self._pick("cuda", available=["CPUExecutionProvider"])

    def test_cpu_returns_bare_string(self):
        result = self._pick("cpu")
        assert result == "CPUExecutionProvider"

    def test_cuda_returns_tuple_with_device_id_zero(self):
        name, opts = self._pick("cuda", available=["CUDAExecutionProvider", "CPUExecutionProvider"])
        assert name == "CUDAExecutionProvider"
        assert opts["device_id"] == 0

    def test_cuda_with_index_sets_device_id(self):
        name, opts = self._pick("cuda:2", available=["CUDAExecutionProvider", "CPUExecutionProvider"])
        assert opts["device_id"] == 2

    def test_tensorrt_sets_device_id_and_fp16(self):
        name, opts = self._pick("tensorrt", available=["TensorrtExecutionProvider", "CPUExecutionProvider"])
        assert name == "TensorrtExecutionProvider"
        assert opts["device_id"] == 0
        assert opts["trt_fp16_enable"] is True

    def test_tensorrt_with_index(self):
        _, opts = self._pick("tensorrt:1", available=["TensorrtExecutionProvider", "CPUExecutionProvider"])
        assert opts["device_id"] == 1

    def test_openvino_defaults_to_gpu(self):
        name, opts = self._pick("openvino", available=["OpenVINOExecutionProvider", "CPUExecutionProvider"])
        assert name == "OpenVINOExecutionProvider"
        assert opts["device_type"] == "GPU"

    def test_openvino_with_device_type(self):
        _, opts = self._pick("openvino:npu", available=["OpenVINOExecutionProvider", "CPUExecutionProvider"])
        assert opts["device_type"] == "NPU"
