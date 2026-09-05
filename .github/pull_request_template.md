## Summary

Please summarize the changes introduced in this pull request and the motivation behind them.

Fixes # (issue)

---

## Type of Change

- [ ] **Bug fix** (non-breaking fix for an issue)
- [ ] **New model / backbone** (e.g. OSNet, ViT-ReID, MobileNetV3)
- [ ] **Loss function / metric learning** (e.g. Circle loss, ArcFace, hard-triplet tuning)
- [ ] **Post-processing** (e.g. $k$-reciprocal encoding, re-ranking)
- [ ] **Evaluation / Benchmark dataset** (e.g. Market-1501, DukeMTMC, MSMT17)
- [ ] **Documentation / Tests** (README, unit test suite, setup instructions)

---

## Testing & Verification

Please confirm the following testing steps have been completed:

- [ ] Dependencies installed via `pip install -r requirements.txt`
- [ ] Automated unit test suite passes: `python3 -m unittest discover -s tests -v`
- [ ] End-to-end evaluation script verified: `python3 main.py`
- [ ] CMC curve (`results/cmc_curve.png`) and metrics (`results/reid_metrics.json`) generated without errors
- [ ] Tested on target device (CPU / CUDA)

---

## Checklist

- [ ] My code adheres to PEP 8 standards.
- [ ] I have documented mathematical formulas with KaTeX syntax.
- [ ] No model checkpoint weights (`.pth`, `.pt`, `.bin`) or temporary datasets are committed.
- [ ] I have updated the README if new features were introduced.
