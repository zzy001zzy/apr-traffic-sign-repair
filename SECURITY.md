# Security policy

The latest 1.0.x release is maintained. Versioned fixes are distributed through GitHub releases; the DOI archive identifies its separately frozen artifact version.

## Report a vulnerability privately

Use GitHub's [Report a vulnerability](https://github.com/zzy001zzy/apr-traffic-sign-repair/security/advisories/new) form. A GitHub account is required. Reports are private to the reporter, invited collaborators and the repository maintainers until an advisory is published. For ordinary reproducible bugs and enhancements, use [GitHub issues](https://github.com/zzy001zzy/apr-traffic-sign-repair/issues).

Include the affected version, a synthetic reproduction, expected impact and any suggested mitigation. Use shareable inputs and keep credentials and private data out of the reproduction. The maintainer aims to acknowledge reports within 7 days, with an initial response within 14 days. Critical reports receive priority triage and a prompt fix or containment measure after confirmation. Confirmed medium or higher severity vulnerabilities are tracked to a versioned correction; public advisories and release notes identify affected versions, fixes, mitigations and assigned CVE/GHSA identifiers.

Public advisories are available on the [repository security page](https://github.com/zzy001zzy/apr-traffic-sign-repair/security/advisories). The release process checks advisories and dependency audit results before publication. As of the initial security-process review on 2026-10-17, the project had received no vulnerability reports and had no published security advisories.

Use Safetensors model files with the included release manifest and verify release SHA-256 values. The public Python interface validates image and mask types, coordinate alignment and model-file integrity. The web example serves static precomputed research panels; local inference operates on files selected by the caller.

## Trust boundary

Obtain both weights and their manifest from the same trusted HTTPS release. SHA-256 detects a changed weight file relative to that manifest; the manifest itself is part of the caller's trust decision. Safetensors avoids executable Python pickle loading. Inference runs locally on caller-selected arrays and model files, with the operating-system permissions of that caller. Deployments accepting untrusted files should add file-size, memory and execution-time limits at their service boundary.

The distributed web example is a static viewer with retained data. It has no project-operated authentication, password database, model-upload endpoint or inference server. Result cells are inserted as text. Hosting TLS is provided by GitHub and Hugging Face. See [`docs/SECURE_DEVELOPMENT.md`](docs/SECURE_DEVELOPMENT.md) for the design review and release checks.
