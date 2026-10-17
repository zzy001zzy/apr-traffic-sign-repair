# Secure development and release process

This document concerns the MIT-licensed APR Python inference library and static viewer. Model weights and the attributed DFG example retain their separate research-asset licenses. The primary maintainer, Chih-Yuan Cheng, confirmed understanding and practical application of the design principles and common-error mitigations below on 2026-10-17.

## Design principles in this project

| Principle | Application |
|---|---|
| Economy of mechanism | A small Python API wraps frozen inference modules; the public viewer serves retained images and aggregates. |
| Fail-safe defaults | Invalid input types, RGB dimensions, mask alignment and manifest matches fail before repair or model loading. |
| Complete mediation | Every public repair call checks its image and mask; both public model loaders check the selected file's manifest entry and SHA-256. |
| Open design | Source, input contracts, weight hashes, licensing and engineering tests are public. Integrity checking relies on a trusted release manifest. |
| Separation of privilege | Release preparation, automatic checks and authenticated repository publication are separate steps; CI has read-only repository permission. |
| Least privilege | CPU tests need no service credentials or GPU. The distributed viewer has no upload or model-execution endpoint. |
| Least common mechanism | Tests use per-test temporary directories; local inference uses caller-owned arrays and returns independent output storage. |
| Psychological acceptability | Input failures describe the expected type, shape or missing integrity property. Installation and verification commands are explicit. |
| Limited attack surface | The static viewer has no project-operated account service or password storage. Local inference exposes three documented functions. |
| Input allowlists | The API accepts RGB uint8 arrays and coordinate-aligned boolean masks. Weights use a tensor-only format with strict state-dictionary loading. |

## Common errors and mitigations

| Risk | Mitigation and verification |
|---|---|
| Executable model deserialization | Load Safetensors, not Python pickle. Check trusted manifest SHA-256 before loading. Tests reject tampering, duplicate/missing entries and malformed tensor files. |
| Command or code injection | The library and static viewer invoke no shell, eval or dynamic source execution. Bandit analyzes all distributed Python inference modules. |
| Cross-site scripting | The viewer inserts result cells through DOM textContent; result data and images are release-owned static assets. |
| Missing authorization | The distributed viewer offers no privileged project server operation. Repository writes and releases use the hosting platform's authenticated maintainer controls. |
| Malformed arrays and unintended mutation | Validate type, RGB dimensions and mask alignment; make owned contiguous copies. Regression tests exercise empty/non-RGB inputs, non-boolean masks and read-only strided arrays. |
| Resource exhaustion | Local callers control input size. Operators adding an untrusted-file service must enforce resource limits at that service boundary. |
| Credential disclosure | Runtime tests need no secrets. CI permissions are read-only; publication review scans the public file tree and Git history for credential-shaped material without printing secret contents. |
| Vulnerable dependencies | Audit the installed CPU runtime and development dependencies in CI with pip-audit. Review results and resolve confirmed exploitable findings before publication. |

SHA-256 comes from Python hashlib and OpenSSL. The project implements no cryptographic algorithm, encryption, key exchange, password storage or cryptographic nonce generation. Package and model delivery uses HTTPS. Third-party hosting provides its own TLS configuration.

## Tests, warnings and analysis

GitHub Actions runs on every push and pull request with repository `contents: read`. It runs Ruff's error, Pyflakes and bugbear rules, all Bandit security checks, the CPU unittest suite under branch-enabled coverage.py, a source-to-wheel build, an installed-dependency audit and REUSE licensing checks. Findings fail their CI steps. Accepted functionality changes include tests in the same pull request; bug fixes include regression cases. New major functionality must have automated tests of valid behavior and relevant failures.

Tests use synthetic inputs and the already released tensor-only model weights. Coverage describes software execution, not benchmark recovery or robustness. A comprehensive fuzzer or separate dynamic security analyzer is a future improvement; coverage-instrumented regression tests run now with unittest assertions enabled. The frozen historical modules remain byte-identical to the source mappings.

## Release procedure

1. Review open bug reports, private reports and public security advisories. Acknowledge ordinary reports and enhancements through GitHub; respond to private vulnerability reports within 14 days, aiming for 7 days. Prioritize critical reports and track confirmed security defects to a correction.
2. Prepare an intermediate commit or pull request with code, tests, documentation, version metadata and release notes. Keep numerical research authorities and frozen model bodies unchanged for engineering-only releases.
3. Run all CI checks against that exact proposed release commit before applying its release tag. Review warnings, Bandit findings, dependency advisories, license inventory and release-content hashes. Fix confirmed exploitable findings before release.
4. Record the version, checked commit, check commands, tool versions and results. Publish a semantic-version Git tag and human-readable GitHub release notes. Identify every CVE/GHSA fixed in the release, when applicable.
5. Archive releases under their own explicit artifact versions when publishing a DOI deposit; retain historical archives and correction notices. An older DOI archive remains identified by its frozen version.

The initial public repository and issue tracker were created on 2026-10-17; at the first security review there were no bug, enhancement or vulnerability reports to acknowledge. This short history is stated explicitly when answering response-history criteria. OpenSSF Best Practices is a voluntary public self-assessment whose justifications link to the repository and checks.

The v1.0.3 dependency review detected upstream PyTorch advisory [CVE-2025-3000 / GHSA-rrmf-rvhw-rf47](https://github.com/advisories/GHSA-rrmf-rvhw-rf47), concerning torch.jit.script. The APR API does not call that function. This release raises its supported PyTorch minimum to the listed fixed version, 2.13.0, and validates its engineering contracts in CPU CI on that version. The build backend minimum is setuptools83, addressing the older build-tool advisories reported by pip-audit. CI audits the installed environment and separately checks the upstream PyTorch version of the official CPU wheel because its +cpu version suffix is not listed on PyPI. Archived benchmark runtimes and measured numbers remain identified separately.
