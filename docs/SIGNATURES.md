<!-- SPDX-FileCopyrightText: 2026 Chih-Yuan Cheng and Bor-Jiunn Hwang -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Verify the signed v1.0.3 release

Release signatures use [Sigstore Cosign](https://docs.sigstore.dev/cosign/signing/signing_with_blobs/)
with a short-lived GitHub Actions OIDC identity. Each bundle contains the
signature, signing certificate and transparency-log verification material.
The signing workflow checks and signs the original published v1.0.3 bytes.

The signature scope is publisher identity and artifact integrity. Research
results and licenses are documented separately in the release.

## Download and verify

Install [Cosign 3.1.3 or a compatible newer version](https://docs.sigstore.dev/cosign/system_config/installation/)
and the [GitHub CLI](https://cli.github.com/). Public release downloads require
no repository write access.

```bash
mkdir apr-v1.0.3-verification
cd apr-v1.0.3-verification
gh release download v1.0.3 --repo zzy001zzy/apr-traffic-sign-repair

identity='https://github.com/zzy001zzy/apr-traffic-sign-repair/.github/workflows/sign-release.yml@refs/heads/codex/public-release'
issuer='https://token.actions.githubusercontent.com'

# Check the signing receipt first, then inspect its artifact hashes and run URL.
cosign verify-blob SIGSTORE-v1.0.3.json \
  --bundle SIGSTORE-v1.0.3.json.sigstore.json \
  --certificate-identity "$identity" --certificate-oidc-issuer "$issuer"

# Verify the original software archive and both released model-weight files.
for artifact in \
  apr-traffic-sign-repair-v1.0.3.zip \
  apr-hardselect96.safetensors \
  roi-presence-mask-cascade.safetensors; do
  cosign verify-blob "$artifact" --bundle "$artifact.sigstore.json" \
    --certificate-identity "$identity" --certificate-oidc-issuer "$issuer"
done

# Check the original release checksums as an additional byte-level comparison.
cosign verify-blob SHA256SUMS.txt --bundle SHA256SUMS.txt.sigstore.json \
  --certificate-identity "$identity" --certificate-oidc-issuer "$issuer"
sha256sum --check SHA256SUMS.txt
```

Cosign checks the exact certificate identity, the GitHub Actions OIDC issuer,
the signature and the transparency evidence. Its certificate can expire after
signing: verification uses the authenticated signing time in the bundle.

The standalone weights are byte-for-byte copies of the corresponding
`weights/` members in the original v1.0.3 ZIP and retain their
[CC-BY-NC-SA-4.0 license](../LICENSES/CC-BY-NC-SA-4.0.txt).
Their expected SHA-256 values are:

| Artifact | SHA-256 |
| --- | --- |
| `apr-hardselect96.safetensors` | `e4f429a7004083b6fa9e228f4431ad4b6ed46fa568b4054fd7b83ddbc1b87d34` |
| `roi-presence-mask-cascade.safetensors` | `bcb5d6c2f267ba850f3745fbb9d605d301bcbdc1b6f2c7078414cb722f633b9c` |

## Maintainer operation

The manually dispatched workflow is restricted to this repository's
`codex/public-release` branch. Its job needs `contents: write` to add release
assets and `id-token: write` to obtain a short-lived signing identity. The
workflow uses pinned action commits and Cosign 3.1.3, validates the original
release ZIP/SPDX/checksum digests and both weight-manifest entries, and checks
every new signature before publishing it.

```bash
gh workflow run sign-release.yml --repo zzy001zzy/apr-traffic-sign-repair \
  --ref codex/public-release -f release_tag=v1.0.3
```

Sigstore's public transparency service records the artifact digest and
certificate identity. `--yes` in the authorized publishing workflow accepts
that public signing operation. GitHub supplies the OIDC token automatically;
the project stores no long-lived signing key.

The upload adds signature bundles, a signing receipt and the original weight
bytes as new release assets. The original ZIP, SPDX and checksum files retain
their existing bytes. Existing asset names cause upload to stop, so a rerun
requires a maintainer to review the published assets first.

See the official [CI signing guide](https://docs.sigstore.dev/quickstart/quickstart-ci/)
and [verification guide](https://docs.sigstore.dev/cosign/verifying/verify/) for
the underlying signature checks.
