# Security policy

Version 1.0 is the current research artifact. Report a suspected vulnerability through the repository's private vulnerability reporting feature when available, or contact the corresponding maintainer through the GitHub profile linked by the repository.

Use Safetensors model files with the included release manifest and verify release SHA-256 values. The public Python interface validates image and mask types, coordinate alignment and model-file integrity. The web example serves static precomputed research panels; local inference operates on files selected by the caller.

Reports should include the affected version, a minimal reproduction and the expected impact. Maintainers will assess reports and document a versioned correction when appropriate.
