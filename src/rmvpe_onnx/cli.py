from .model import RMVPE
from .weights import ensure_model


def _plot(time, f0, confidence, activation):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("Package 'plotly' is required for plotting. Install with 'pip install rmvpe-onnx[cli]'") from None

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.4, 0.3, 0.3],
        vertical_spacing=0.06,
        subplot_titles=("Activation Heatmap", "F0 (Hz)", "Confidence"),
    )

    # ── activation heatmap ───────────────────────────────────────────────────
    fig.add_trace(
        go.Heatmap(
            x=time,
            z=activation.T,                     # (360, T)
            colorscale="Turbo",
            showscale=False,
            name="activation",
        ),
        row=1, col=1,
    )

    # ── F0 ───────────────────────────────────────────────────────────────────
    voiced_mask   = f0 > 0
    unvoiced_mask = ~voiced_mask

    fig.add_trace(
        go.Scatter(
            x=time[unvoiced_mask], y=confidence[unvoiced_mask],
            mode="markers",
            marker={"color": "#555555", "size": 2, "symbol": "line-ns-open"},
            name="unvoiced",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time[voiced_mask], y=f0[voiced_mask],
            mode="markers",
            marker={
                "color": confidence[voiced_mask],
                "colorscale": "Turbo",
                "cmin": 0, "cmax": 1,
                "size": 3,
                "showscale": True,
                "colorbar": {"title": "conf", "thickness": 12, "x": 1.02, "len": 0.3, "y": 0.17},
            },
            name="F0",
        ),
        row=2, col=1,
    )

    # ── confidence ───────────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=time, y=confidence,
            mode="lines",
            line={"color": "#7eb3d8", "width": 1},
            name="confidence",
            fill="tozeroy",
            fillcolor="rgba(126,179,216,0.2)",
        ),
        row=3, col=1,
    )

    voiced = f0[voiced_mask]
    fig.update_layout(
        title={
            "text": (
                f"{len(f0)} frames  ·  "
                f"{100 * voiced_mask.sum() / len(f0):.1f} % voiced  ·  "
                f"F0 {voiced.min():.0f}–{voiced.max():.0f} Hz"
                if voiced.size else f"{len(f0)} frames  ·  0 % voiced"
            ),
            "font": {"size": 13},
        },
        template="plotly_dark",
        height=700,
        showlegend=True,
        legend={"orientation": "h", "y": -0.08},
        margin={"l": 60, "r": 80, "t": 70, "b": 60},
    )
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Cent bin",     row=1, col=1)
    fig.update_yaxes(title_text="F0 (Hz)",      row=2, col=1)
    fig.update_yaxes(title_text="Confidence",   row=3, col=1, range=[0, 1])

    fig.show()


def get_parser():
    """Return the argument parser for the ``rmvpe-onnx`` CLI.

    Used by Sphinx ``sphinxarg.ext`` to auto-generate CLI documentation.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="rmvpe-onnx",
        description="RMVPE pitch estimator — ONNX Runtime inference",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── download ─────────────────────────────────────────────────────────────
    dl = sub.add_parser("download", help="Download the ONNX model")
    dl.add_argument("--model", default=None, help="Destination path for rmvpe.onnx (default: default model path)")  # noqa: E501

    # ── predict ──────────────────────────────────────────────────────────────
    run = sub.add_parser("predict", help="Run pitch prediction on an audio file")
    run.add_argument("audio",                                        help="Path to input audio file (any format supported by soundfile, typically WAV)")  # noqa: E501
    run.add_argument("--model",               default=None,          help="Path to rmvpe.onnx (default: default model path)")  # noqa: E501
    run.add_argument("--device",              default=None,          help="Inference device to use: cpu, cuda, cuda:N, dml, or coreml (default: auto-detect)")  # noqa: E501
    run.add_argument("--csv",                 default=None,          help="Save results to a CSV file (columns: time, frequency, confidence); threshold is applied before saving")  # noqa: E501
    run.add_argument("--confidence-threshold", type=float, default=0.03, dest="confidence_threshold", help="Frames below this confidence threshold (0.0–1.0) are zeroed out (default: 0.03)")  # noqa: E501
    run.add_argument("--plot",                action="store_true",   help="Show an interactive pitch plot after inference")  # noqa: E501

    return parser


def main():
    import os
    import time

    import numpy as np

    args = get_parser().parse_args()

    # ── download command ──────────────────────────────────────────────────────
    if args.command == "download":
        download_path = ensure_model(args.model)
        print(f"Model ready: {os.path.abspath(download_path)}")
        return

    # ── predict command ───────────────────────────────────────────────────────
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("Package 'soundfile' is required. Install with 'pip install rmvpe-onnx[cli]'") from None

    if not os.path.isfile(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    audio, sr = sf.read(args.audio, always_2d=False)

    rmvpe = RMVPE(model_path=args.model, device=args.device)

    t0 = time.perf_counter()
    timestamp, f0, confidence, activation = rmvpe.predict(audio, sr)
    elapsed = time.perf_counter() - t0

    if args.confidence_threshold is not None:
        f0[confidence < args.confidence_threshold] = 0

    voiced = f0[f0 > 0]
    print(f"Audio      : {os.path.abspath(args.audio)}")
    print(f"Model      : {os.path.abspath(rmvpe.model_path)}")
    print(f"Provider   : {rmvpe.session.get_providers()[0]}")
    print(f"Threshold  : {args.confidence_threshold:.2f}")
    print(f"Frames     : {len(f0)}")
    print(f"Voiced     : {len(voiced)}  ({100 * len(voiced) / len(f0):.1f} %)")
    if voiced.size:
        print(f"F0 range   : {voiced.min():.1f} – {voiced.max():.1f} Hz")
        print(f"F0 median  : {np.median(voiced):.1f} Hz")
    print(f"Time       : {elapsed * 1000:.1f} ms")

    if args.csv:
        np.savetxt(args.csv, np.vstack([timestamp, f0, confidence]).T,
                   fmt=["%.3f", "%.3f", "%.6f"], delimiter=",",
                   header="time,frequency,confidence", comments="")
        print(f"Saved CSV  → {os.path.abspath(args.csv)}")

    if args.plot:
        _plot(timestamp, f0, confidence, activation)


if __name__ == "__main__":  # pragma: no cover
    main()
