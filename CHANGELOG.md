# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog,
and this project adheres to Semantic Versioning.

<!-- version list -->

## v0.2.2 (2026-04-10)

### Bug Fixes

- Relax numpy lower bound to support numpy 1.x users
  ([`4ce158e`](https://github.com/NewComer00/rmvpe-onnx/commit/4ce158e594096d47f47df58b03874705f3d060e8))


## v0.2.1 (2026-04-10)

### Bug Fixes

- Tighten version markers for numpy and onnxruntime on Python 3.10
  ([`a542dde`](https://github.com/NewComer00/rmvpe-onnx/commit/a542dde8cabbeb9e280dbb510de254f1fabb10bd))

### Documentation

- Add uv install instructions and development setup
  ([`77856f1`](https://github.com/NewComer00/rmvpe-onnx/commit/77856f1a240bdc72e3ae3be9f6a806f5529615e2))


## v0.2.0 (2026-04-10)

### Features

- Add Python 3.10 support and tox-based CI pipeline
  ([`e334fba`](https://github.com/NewComer00/rmvpe-onnx/commit/e334fba1ecdfef59abf0b7ca75a1294e8528838d))

- Add Python 3.14 to the CI matrix
  ([`7200e53`](https://github.com/NewComer00/rmvpe-onnx/commit/7200e534be70539b27895eab4e45f13de7aa5dac))


## v0.1.0 (2026-04-09)

### Bug Fixes

- Rename --confidence_threshold to --confidence-threshold
  ([`34d424a`](https://github.com/NewComer00/rmvpe-onnx/commit/34d424adcd36f3eda65006bf2efbc339c79c50bd))

### Build System

- Bump minimum Python to 3.11 and onnxruntime to 1.24
  ([`2fa808c`](https://github.com/NewComer00/rmvpe-onnx/commit/2fa808ca6880d25979503d483521d2ce9bce1e32))

### Chores

- Update the version of softprops/action-gh-release
  ([`40d18fb`](https://github.com/NewComer00/rmvpe-onnx/commit/40d18fb9693e32024b1e7300956e619d204606fd))

### Continuous Integration

- Add macOS to test matrix and upload coverage from all platforms
  ([`34d424a`](https://github.com/NewComer00/rmvpe-onnx/commit/34d424adcd36f3eda65006bf2efbc339c79c50bd))

### Documentation

- Simplify README and add Sphinx autosummary template
  ([`34d424a`](https://github.com/NewComer00/rmvpe-onnx/commit/34d424adcd36f3eda65006bf2efbc339c79c50bd))

### Features

- **weights**: Add SHA-256 verification for the ONNX model file
  ([`34d424a`](https://github.com/NewComer00/rmvpe-onnx/commit/34d424adcd36f3eda65006bf2efbc339c79c50bd))


## v0.0.0 (2026-04-09)

- Initial Release
