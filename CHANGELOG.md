# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-03-21

### Added
- Initial release of CSM Gradio UI
- Support for CUDA, MLX (Apple Silicon), and CPU backends
- Interactive web interface with voice cloning capabilities
- Real-time generation statistics display
- Automatic backend selection based on hardware
- Custom voice prompt support for each speaker
- Memory and timing metrics for MLX backend

### Requirements
- VRAM requirement: 8.1 GB for model loading
- Python 3.10 or higher
- Dependencies listed in requirements.txt

### Known Issues
- MLX dependencies only install on Apple Silicon devices
- Memory metrics only available for MLX backend

## [0.1.0] - 2025-03-15

### Added
- Basic CPU support
- Initial Gradio UI implementation

## [0.0.1] - 2025-03-13

### Added
- Initial release of CSM 1B model
- Basic PyTorch implementation 