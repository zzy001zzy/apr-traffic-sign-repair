# SPDX-FileCopyrightText: 2026 Chih-Yuan Cheng and Bor-Jiunn Hwang
# SPDX-License-Identifier: MIT
"""Prepare exact published v1.0.3 bytes for keyless Sigstore signing."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path


REPOSITORY = "zzy001zzy/apr-traffic-sign-repair"
TAG = "v1.0.3"
IDENTITY = (
    f"https://github.com/{REPOSITORY}/.github/workflows/sign-release.yml"
    "@refs/heads/codex/public-release"
)
ISSUER = "https://token.actions.githubusercontent.com"
ASSETS = {
    "apr-traffic-sign-repair-v1.0.3.zip": (
        "7d22f0920c9ac6407582fea379f891f3c95abf8fc0330ffacad7e370ee85d09b",
        1_147_309,
    ),
    "apr-traffic-sign-repair-v1.0.3.spdx": (
        "ffd40435817f4dcd9e49789fdd1167693e14c2bbded18de8ae3503db005f73e3",
        15_346,
    ),
    "SHA256SUMS.txt": (
        "e9a614a6f52599637b69dee76c3fbd8eb3e4f3452be04a9b5f6a34c8721638b5",
        203,
    ),
}
WEIGHTS = {
    "apr-hardselect96.safetensors": (
        "e4f429a7004083b6fa9e228f4431ad4b6ed46fa568b4054fd7b83ddbc1b87d34",
        112_832,
    ),
    "roi-presence-mask-cascade.safetensors": (
        "bcb5d6c2f267ba850f3745fbb9d605d301bcbdc1b6f2c7078414cb722f633b9c",
        751_444,
    ),
}


def checked_bytes(path: Path, expected: tuple[str, int]) -> bytes:
    """Read an artifact only if its published size and digest both match."""
    digest, size = expected
    if path.stat().st_size != size:
        raise ValueError(f"Unexpected byte count: {path.name}")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f"Unexpected SHA-256: {path.name}")
    return data


def prepare(args: argparse.Namespace) -> dict:
    if args.tag != TAG:
        raise ValueError(f"Only the reviewed published {TAG} release is supported")
    if not re.fullmatch(r"[0-9a-f]{40}", args.workflow_sha):
        raise ValueError("workflow-sha must be a full lowercase Git commit SHA")
    if not args.run_id.isdecimal() or not args.run_attempt.isdecimal():
        raise ValueError("GitHub run identifiers must be decimal integers")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Output directory must be empty")

    originals = {
        name: checked_bytes(args.input_dir / name, expected)
        for name, expected in ASSETS.items()
    }
    expected_sums = "".join(
        f"{ASSETS[name][0]}  {name}\n"
        for name in ASSETS
        if name != "SHA256SUMS.txt"
    )
    if originals["SHA256SUMS.txt"].decode("ascii") != expected_sums:
        raise ValueError("Published checksum list disagrees with pinned assets")

    prefix = "apr-traffic-sign-repair-1.0.3/weights/"
    extracted = {}
    with zipfile.ZipFile(args.input_dir / "apr-traffic-sign-repair-v1.0.3.zip") as archive:
        manifest = json.loads(archive.read(f"{prefix}manifest.json"))
        if {entry["file"] for entry in manifest} != set(WEIGHTS):
            raise ValueError("Weight manifest does not contain exactly the reviewed weights")
        for name, (digest, size) in WEIGHTS.items():
            entry = next(item for item in manifest if item["file"] == name)
            if (
                entry["sha256"] != digest
                or entry["bytes"] != size
                or entry["license"] != "CC-BY-NC-SA-4.0"
            ):
                raise ValueError(f"Weight manifest mismatch: {name}")
            info = archive.getinfo(f"{prefix}{name}")
            if info.file_size != size:
                raise ValueError(f"Unexpected archived weight size: {name}")
            data = archive.read(info)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Unexpected archived weight digest: {name}")
            extracted[name] = data

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ASSETS:
        shutil.copyfile(args.input_dir / name, args.output_dir / name)
        checked_bytes(args.output_dir / name, ASSETS[name])
    for name, data in extracted.items():
        (args.output_dir / name).write_bytes(data)

    metadata = {
        "schema_version": 1,
        "release_tag": TAG,
        "release_url": f"https://github.com/{REPOSITORY}/releases/tag/{TAG}",
        "signing_method": "Sigstore Cosign keyless blob signatures with GitHub Actions OIDC",
        "certificate_identity": IDENTITY,
        "certificate_oidc_issuer": ISSUER,
        "signing_workflow_commit": args.workflow_sha,
        "signing_run_url": f"https://github.com/{REPOSITORY}/actions/runs/{args.run_id}",
        "signing_run_attempt": int(args.run_attempt),
        "signed_artifacts": [
            {
                "file": name,
                "sha256": digest,
                "bytes": size,
                "bundle": f"{name}.sigstore.json",
                "source": "original release asset" if name in ASSETS else "original release ZIP weights/ member",
                **({"license": "CC-BY-NC-SA-4.0"} if name in WEIGHTS else {}),
            }
            for name, (digest, size) in {**ASSETS, **WEIGHTS}.items()
        ],
        "scope": "Publisher identity and exact artifact integrity",
    }
    receipt_name = f"SIGSTORE-{TAG}.json"
    (args.output_dir / receipt_name).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    targets = [*ASSETS, *WEIGHTS, receipt_name]
    (args.output_dir / "signing-inputs.txt").write_text(
        "\n".join(targets) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workflow-sha", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-attempt", required=True)
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
