# Contributing to Multi-Camera Person Re-Identification (ReID)

Thank you for your interest in contributing to the **Multi-Camera Person Re-Identification (ReID) via Deep Feature Embedding** project! We welcome contributions from researchers, computer vision engineers, and open-source contributors.

Please read through this guide before submitting issues or pull requests.

---

## Code of Conduct

All contributors and participants must adhere to our [Code of Conduct](CODE_OF_CONDUCT.md). Please report any violations or inappropriate behavior to the project maintainer.

---

## Ways to Contribute

We welcome contributions across several core areas of Person Re-ID:

- **Backbone Architectures**: Integrate lightweight edge models (OSNet, MobileNetV3, ShuffleNet) or transformer-based models (TransReID, ViT-ReID).
- **Metric Learning & Loss Functions**: Implement advanced losses such as Circle Loss, ArcFace, Center Loss, or Supervised Contrastive Loss.
- **Post-Processing & Re-Ranking**: Implement $k$-reciprocal feature encoding or camera-aware re-ranking algorithms to improve Rank-1 accuracy.
- **Dataset Support**: Add loaders and evaluation scripts for public ReID benchmarks (Market-1501, DukeMTMC-reID, MSMT17, CUHK03).
- **Occlusion Handling & Part-Based Models**: Add PCB (Part-based Convolutional Baseline) or attention-based occlusion-aware modules.
- **Inference Optimization**: Add FP16 half-precision support, ONNX export, TensorRT engine compilation, or batch inference profiling.

---

## Reporting Issues & Bugs

Before submitting an issue, please check existing [GitHub Issues](https://github.com/gbhanuvigneshnaidu29052002-droid/person-re-identification/issues) to avoid duplicates.

When filing a bug report:
1. Use our [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md).
2. Specify your hardware & OS environment:
   - OS: Ubuntu 22.04 / Linux / Windows / macOS
   - Python Version: Python 3.10+
   - PyTorch Version: PyTorch 2.x
   - Device: CPU, CUDA (with GPU model and driver version)
3. Include minimal code or terminal commands to reproduce the error.
4. Attach complete stack traces and logs.

---

## Development Workflow

### 1. Fork & Clone Repository
```bash
git clone https://github.com/<your-username>/person-re-identification.git
cd person-re-identification
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Create a Feature Branch
```bash
git checkout -b feature/k-reciprocal-re-ranking
# or
git checkout -b fix/cosine-distance-matrix-shape
```

### 4. Run Automated Unit Tests
Verify that all unit tests pass:
```bash
python3 -m unittest discover -s tests -v
```

### 5. Run Benchmark CLI
Verify end-to-end evaluation, metrics export, and plot generation:
```bash
python3 main.py --backbone resnet50 --output_dir results
```

---

## Code Quality Standards

- **Python Style**: Follow PEP 8 guidelines.
- **Type Annotations**: Include type hints and clear docstrings for public classes and functions.
- **Device Agnostic**: Ensure code runs seamlessly on both `cpu` and `cuda`.
- **Git Hygiene**: Keep commits focused and atomic. Never commit large model weights (`.pth`, `.pt`, `.onnx`), virtual environments, or temporary dataset caches.

---

## Submitting a Pull Request

1. Fill out the [Pull Request Template](.github/pull_request_template.md).
2. Confirm that all automated unit tests pass (`python3 -m unittest discover -s tests -v`).
3. Include before/after benchmark metrics (Rank-1, Rank-5, Rank-10, mAP) in the PR description if modifying model architectures or loss functions.

Thank you for helping push computer vision and surveillance research forward!
