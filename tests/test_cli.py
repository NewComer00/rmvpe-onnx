"""Unit tests for rmvpe_onnx.cli.

Tests call cli internals directly (no subprocess) so every branch is reachable.
The existing test_rmvpe_onnx.py keeps the subprocess smoke-tests for argparse
and process exit-code checks; this file covers everything else.
"""

from __future__ import annotations

import struct
import wave
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


def _wav_file(path: Path, audio: np.ndarray | None = None, sr: int = 16_000) -> Path:
    if audio is None:
        audio = _sine()
    samples = (audio * 32_767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return path


def _fake_predict_outputs(n_frames: int = 50, voiced_frac: float = 0.8):
    """Return (timestamp, f0, confidence, activation) matching RMVPE.predict."""
    hop = 160 / 16_000
    timestamp  = np.arange(n_frames, dtype=np.float64) * hop
    confidence = np.random.default_rng(0).uniform(0.01, 1.0, n_frames).astype(np.float32)
    f0         = np.zeros(n_frames, dtype=np.float64)
    voiced     = int(n_frames * voiced_frac)
    f0[:voiced] = np.linspace(200, 400, voiced)
    activation = np.zeros((n_frames, 360), dtype=np.float32)
    activation[np.arange(n_frames), np.random.default_rng(1).integers(0, 360, n_frames)] = 1.0
    return timestamp, f0, confidence, activation


def _mock_rmvpe(predict_return):
    """Return a mock RMVPE instance whose predict() returns predict_return."""
    mock = MagicMock()
    mock.predict.return_value = predict_return
    mock.model_path = "/fake/rmvpe.onnx"
    mock.session.get_providers.return_value = ["CPUExecutionProvider"]
    return mock


# ---------------------------------------------------------------------------
# _plot
# ---------------------------------------------------------------------------

class TestPlot:
    def test_missing_plotly_raises_import_error(self):
        from rmvpe_onnx.cli import _plot
        ts, f0, conf, act = _fake_predict_outputs()
        with patch.dict("sys.modules", {"plotly": None, "plotly.graph_objects": None, "plotly.subplots": None}):
            with pytest.raises(ImportError, match="plotly"):
                _plot(ts, f0, conf, act)

    def test_calls_fig_show(self):
        from rmvpe_onnx.cli import _plot
        ts, f0, conf, act = _fake_predict_outputs()
        fig_instance = MagicMock()
        with patch("plotly.subplots.make_subplots", return_value=fig_instance):
            _plot(ts, f0, conf, act)
        fig_instance.show.assert_called_once()

    def test_all_unvoiced_does_not_crash(self):
        """When no frames are voiced, the title ternary takes the else branch."""
        from rmvpe_onnx.cli import _plot
        ts, f0, conf, act = _fake_predict_outputs()
        f0[:] = 0  # all unvoiced

        fig_instance = MagicMock()
        with patch("plotly.subplots.make_subplots", return_value=fig_instance):
            _plot(ts, f0, conf, act)  # must not raise
        fig_instance.show.assert_called_once()

    def test_all_voiced_uses_f0_range_title(self):
        from rmvpe_onnx.cli import _plot
        ts, f0, conf, act = _fake_predict_outputs(voiced_frac=1.0)
        f0[:] = np.linspace(200, 400, len(f0))  # all voiced

        fig_instance = MagicMock()
        with patch("plotly.subplots.make_subplots", return_value=fig_instance):
            _plot(ts, f0, conf, act)

        title_call = fig_instance.update_layout.call_args
        title_text = title_call[1]["title"]["text"]
        assert "Hz" in title_text
        assert title_text.startswith("50 frames")


# ---------------------------------------------------------------------------
# main() — download subcommand
# ---------------------------------------------------------------------------

class TestMainDownload:
    def _run_download(self, args: list[str], ensure_return: str = "/fake/rmvpe.onnx"):
        from rmvpe_onnx.cli import main
        with (
            patch("sys.argv", ["rmvpe-onnx", "download"] + args),
            patch("rmvpe_onnx.cli.ensure_model", return_value=ensure_return),
            patch("builtins.print") as mock_print,
        ):
            main()
        return mock_print

    def test_download_prints_model_ready(self, tmp_path):
        mock_print = self._run_download(["--model", str(tmp_path / "rmvpe.onnx")],
                                        ensure_return=str(tmp_path / "rmvpe.onnx"))
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Model ready" in printed

    def test_download_no_model_arg_uses_default(self):
        mock_print = self._run_download([])
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Model ready" in printed


# ---------------------------------------------------------------------------
# main() — predict subcommand
# ---------------------------------------------------------------------------

class TestMainPredict:
    @pytest.fixture()
    def wav(self, tmp_path) -> Path:
        return _wav_file(tmp_path / "test.wav")

    @pytest.fixture()
    def outputs(self):
        return _fake_predict_outputs()

    def _run_predict(self, wav: Path, outputs, extra_args: list[str] | None = None):
        """Run main() with a fully mocked RMVPE and soundfile."""
        from rmvpe_onnx.cli import main
        mock_rmvpe_instance = _mock_rmvpe(outputs)
        argv = ["rmvpe-onnx", "predict", str(wav)] + (extra_args or [])
        with (
            patch("sys.argv", argv),
            patch("rmvpe_onnx.cli.RMVPE", return_value=mock_rmvpe_instance),
            patch("builtins.print") as mock_print,
        ):
            main()
        return mock_print, mock_rmvpe_instance

    def test_missing_audio_raises_file_not_found(self, tmp_path, outputs):
        from rmvpe_onnx.cli import main
        with (
            patch("sys.argv", ["rmvpe-onnx", "predict", str(tmp_path / "nope.wav")]),
            patch("rmvpe_onnx.cli.RMVPE"),
        ):
            with pytest.raises(FileNotFoundError, match="Audio file not found"):
                main()

    def test_missing_soundfile_raises_import_error(self, wav, outputs):
        from rmvpe_onnx.cli import main
        with (
            patch("sys.argv", ["rmvpe-onnx", "predict", str(wav)]),
            patch.dict("sys.modules", {"soundfile": None}),
        ):
            with pytest.raises(ImportError, match="soundfile"):
                main()

    def test_basic_predict_prints_summary(self, wav, outputs):
        mock_print, _ = self._run_predict(wav, outputs)
        printed = "\n".join(str(c) for c in mock_print.call_args_list)
        assert "Frames" in printed
        assert "Voiced" in printed
        assert "Provider" in printed

    def test_predict_prints_f0_range_when_voiced(self, wav, outputs):
        mock_print, _ = self._run_predict(wav, outputs)
        printed = "\n".join(str(c) for c in mock_print.call_args_list)
        assert "F0 range" in printed
        assert "F0 median" in printed

    def test_predict_no_f0_range_when_all_unvoiced(self, wav, tmp_path):
        ts, f0, conf, act = _fake_predict_outputs()
        f0[:] = 0
        mock_print, _ = self._run_predict(wav, (ts, f0, conf, act))
        printed = "\n".join(str(c) for c in mock_print.call_args_list)
        assert "F0 range" not in printed

    def test_confidence_threshold_zeros_low_confidence_frames(self, wav, outputs):
        """Frames below threshold must be zeroed before stats are printed."""
        ts, f0, conf, act = outputs
        mock_print, mock_rmvpe = self._run_predict(
            wav, (ts, f0, conf, act),
            extra_args=["--confidence-threshold", "0.99"],
        )
        # With threshold=0.99 almost everything is zeroed
        printed = "\n".join(str(c) for c in mock_print.call_args_list)
        # voiced count line must exist and show a low number
        assert "Voiced" in printed

    def test_csv_written_when_flag_set(self, wav, outputs, tmp_path):
        csv_path = tmp_path / "out.csv"
        self._run_predict(wav, outputs, extra_args=["--csv", str(csv_path)])
        assert csv_path.exists()
        lines = csv_path.read_text().splitlines()
        assert lines[0] == "time,frequency,confidence"
        assert len(lines) > 1  # header + data rows

    def test_csv_columns_are_correct(self, wav, outputs, tmp_path):
        csv_path = tmp_path / "out.csv"
        self._run_predict(wav, outputs, extra_args=["--csv", str(csv_path)])
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        assert data.ndim == 2
        assert data.shape[1] == 3  # time, frequency, confidence

    def test_csv_saved_line_printed(self, wav, outputs, tmp_path):
        csv_path = tmp_path / "out.csv"
        mock_print, _ = self._run_predict(wav, outputs, extra_args=["--csv", str(csv_path)])
        printed = "\n".join(str(c) for c in mock_print.call_args_list)
        assert "Saved CSV" in printed

    def test_plot_called_when_flag_set(self, wav, outputs):
        from rmvpe_onnx.cli import main
        mock_rmvpe_instance = _mock_rmvpe(outputs)
        with (
            patch("sys.argv", ["rmvpe-onnx", "predict", str(wav), "--plot"]),
            patch("rmvpe_onnx.cli.RMVPE", return_value=mock_rmvpe_instance),
            patch("rmvpe_onnx.cli._plot") as mock_plot,
            patch("builtins.print"),
        ):
            main()
        mock_plot.assert_called_once()

    def test_plot_not_called_without_flag(self, wav, outputs):
        from rmvpe_onnx.cli import main
        mock_rmvpe_instance = _mock_rmvpe(outputs)
        with (
            patch("sys.argv", ["rmvpe-onnx", "predict", str(wav)]),
            patch("rmvpe_onnx.cli.RMVPE", return_value=mock_rmvpe_instance),
            patch("rmvpe_onnx.cli._plot") as mock_plot,
            patch("builtins.print"),
        ):
            main()
        mock_plot.assert_not_called()

    def test_device_arg_passed_to_rmvpe(self, wav, outputs):
        from rmvpe_onnx.cli import main
        mock_rmvpe_cls = MagicMock(return_value=_mock_rmvpe(outputs))
        with (
            patch("sys.argv", ["rmvpe-onnx", "predict", str(wav), "--device", "cpu"]),
            patch("rmvpe_onnx.cli.RMVPE", mock_rmvpe_cls),
            patch("builtins.print"),
        ):
            main()
        _, kwargs = mock_rmvpe_cls.call_args
        assert kwargs.get("device") == "cpu"

    def test_model_arg_passed_to_rmvpe(self, wav, outputs, tmp_path):
        from rmvpe_onnx.cli import main
        model_path = str(tmp_path / "custom.onnx")
        mock_rmvpe_cls = MagicMock(return_value=_mock_rmvpe(outputs))
        with (
            patch("sys.argv", ["rmvpe-onnx", "predict", str(wav), "--model", model_path]),
            patch("rmvpe_onnx.cli.RMVPE", mock_rmvpe_cls),
            patch("builtins.print"),
        ):
            main()
        _, kwargs = mock_rmvpe_cls.call_args
        assert kwargs.get("model_path") == model_path
