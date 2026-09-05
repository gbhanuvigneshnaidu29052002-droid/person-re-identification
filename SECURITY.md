# Security Policy

## Supported Versions

Security fixes and updates are provided for the following releases:

| Version / Branch | Python Version | PyTorch | Status |
| :--- | :--- | :--- | :--- |
| `main` | 3.10+ | 2.0+ | :white_check_mark: Supported |
| Older commits | < 3.10 | < 2.0 | :x: Not supported |

---

## Scope & Ethical Threat Model

In person re-identification, video surveillance, and biometric representation learning systems, security and privacy constraints are paramount:

- **Adversarial Pattern Exploits**: Susceptibility to adversarial clothing patches, physical camouflage, or noise perturbations designed to fool re-identification matching.
- **Model Checkpoint Integrity**: Unsafe deserialization of PyTorch `.pth` or `.pt` model checkpoints. Always verify SHA256 checksums or use safe model loading conventions.
- **Privacy & Data Security**: Ensuring test/benchmark datasets do not contain unauthorized personally identifiable information (PII) or unredacted facial identities without consent.
- **Denial-of-Service**: Malformed high-resolution image inputs causing memory exhaustion in feature extraction loops.

---

## Reporting a Vulnerability

If you discover a security vulnerability or sensitive data issue in this project, please report it responsibly:

### How to Report

1. **GitHub Security Advisory (Preferred)**:
   - Go to the repository's **Security** tab.
   - Click **Report a vulnerability** to start a confidential draft advisory.
2. **Direct Maintainer Contact**:
   - Reach out to the maintainer via GitHub: [@gbhanuvigneshnaidu29052002-droid](https://github.com/gbhanuvigneshnaidu29052002-droid).

### What to Include

- Detailed description of the vulnerability and attack vector.
- Minimal reproducible example or proof-of-concept (PoC).
- Proposed mitigation or remediation steps.

### Response Commitment

- Acknowledgment within 48 hours.
- Triage and remediation timeline provided promptly.
- Public attribution upon patch publication (unless anonymity is requested).

Please refrain from submitting public issues for unresolved security concerns.
