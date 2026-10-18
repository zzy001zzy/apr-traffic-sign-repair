# SPDX-FileCopyrightText: 2026 Chih-Yuan Cheng and Bor-Jiunn Hwang
# SPDX-License-Identifier: MIT
"""Exercise signing preparation with locally generated synthetic release bytes."""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import zipfile

from scripts import prepare_release_signatures as signing


def digest_size(data):
    return hashlib.sha256(data).hexdigest(), len(data)


class ReleaseSignatureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.output = self.root / "output"
        self.payloads = {"repair.safetensors": b"synthetic repair", "locator.safetensors": b"synthetic locator"}
        self.weights = {name: digest_size(data) for name, data in self.payloads.items()}
        self.manifest = [{"file": name, "sha256": digest, "bytes": size, "license": "CC-BY-NC-SA-4.0"}
                         for name, (digest, size) in self.weights.items()]
        self.assets = {}
        self.write_release()
        patcher = mock.patch.multiple(signing, ASSETS=self.assets, WEIGHTS=self.weights)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.args = argparse.Namespace(tag=signing.TAG, input_dir=self.inputs, output_dir=self.output,
                                       workflow_sha="a" * 40, run_id="123", run_attempt="1")

    def write_release(self):
        prefix = "apr-traffic-sign-repair-1.0.3/weights/"
        archive = self.inputs / "apr-traffic-sign-repair-v1.0.3.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr(prefix + "manifest.json", json.dumps(self.manifest))
            for name, payload in self.payloads.items():
                stream.writestr(prefix + name, payload)
        spdx = self.inputs / "apr-traffic-sign-repair-v1.0.3.spdx"
        spdx.write_bytes(b"Synthetic SPDX test fixture\n")
        self.assets.clear()
        self.assets.update({path.name: digest_size(path.read_bytes()) for path in (archive, spdx)})
        sums = "".join(f"{digest}  {name}\n" for name, (digest, _) in self.assets.items())
        sums_path = self.inputs / "SHA256SUMS.txt"
        sums_path.write_text(sums, encoding="ascii")
        self.assets[sums_path.name] = digest_size(sums_path.read_bytes())

    def test_success_preserves_every_asset_and_records_exact_identity(self):
        receipt = signing.prepare(self.args)
        for name in self.assets:
            self.assertEqual((self.output / name).read_bytes(), (self.inputs / name).read_bytes())
        for name, payload in self.payloads.items():
            self.assertEqual((self.output / name).read_bytes(), payload)
        self.assertEqual(receipt["certificate_identity"], signing.IDENTITY)
        self.assertEqual(receipt["certificate_oidc_issuer"], signing.ISSUER)
        self.assertEqual(receipt["signing_workflow_commit"], self.args.workflow_sha)
        self.assertEqual(len(receipt["signed_artifacts"]), 5)
        saved = json.loads((self.output / f"SIGSTORE-{signing.TAG}.json").read_text())
        self.assertEqual(saved, receipt)
        targets = (self.output / "signing-inputs.txt").read_text().splitlines()
        self.assertEqual(targets, [*self.assets, *self.weights, f"SIGSTORE-{signing.TAG}.json"])

    def test_checked_bytes_rejects_size_digest_and_missing_files(self):
        path = self.inputs / "SHA256SUMS.txt"
        expected = self.assets[path.name]
        with self.assertRaisesRegex(ValueError, "byte count"):
            signing.checked_bytes(path, (expected[0], expected[1] + 1))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            signing.checked_bytes(path, ("0" * 64, expected[1]))
        with self.assertRaises(FileNotFoundError):
            signing.checked_bytes(self.inputs / "missing", expected)

    def test_unreviewed_tag_and_invalid_run_identifiers_fail_before_output(self):
        for changes in ({"tag": "v1.0.4"}, {"workflow_sha": "abc"}, {"workflow_sha": "A" * 40},
                        {"run_id": "123;echo"}, {"run_attempt": "-1"}):
            args = argparse.Namespace(**(vars(self.args) | changes))
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    signing.prepare(args)
                self.assertFalse(self.output.exists())

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir()
        existing = self.output / "existing.txt"
        existing.write_bytes(b"keep")
        with self.assertRaisesRegex(ValueError, "empty"):
            signing.prepare(self.args)
        self.assertEqual(existing.read_bytes(), b"keep")

    def test_disagreeing_checksum_list_is_rejected(self):
        path = self.inputs / "SHA256SUMS.txt"
        path.write_bytes(b"incorrect checksum list\n")
        self.assets[path.name] = digest_size(path.read_bytes())
        with self.assertRaisesRegex(ValueError, "checksum list"):
            signing.prepare(self.args)
        self.assertFalse(self.output.exists())

    def test_manifest_requires_exact_weight_inventory_and_metadata(self):
        reviewed = [dict(entry) for entry in self.manifest]
        for field, invalid in (("file", "unexpected.safetensors"), ("sha256", "0" * 64),
                               ("bytes", 1), ("license", "MIT")):
            self.manifest = [dict(entry) for entry in reviewed]
            self.manifest[0][field] = invalid
            self.write_release()
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "manifest"):
                    signing.prepare(self.args)
                self.assertFalse(self.output.exists())

    def test_archive_weight_size_and_digest_are_checked_independently(self):
        name = next(iter(self.payloads))
        original = self.payloads[name]
        for payload, message in ((b"short", "size"), (b"x" * len(original), "digest")):
            self.payloads[name] = payload
            self.write_release()
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    signing.prepare(self.args)
                self.assertFalse(self.output.exists())

    def test_command_line_main_emits_the_saved_receipt(self):
        argv = ["prepare_release_signatures.py", "--tag", signing.TAG,
                "--input-dir", str(self.inputs), "--output-dir", str(self.output),
                "--workflow-sha", self.args.workflow_sha, "--run-id", "123", "--run-attempt", "1"]
        output = io.StringIO()
        with mock.patch("sys.argv", argv), contextlib.redirect_stdout(output):
            signing.main()
        receipt = json.loads(output.getvalue())
        self.assertEqual(receipt["release_tag"], signing.TAG)
        self.assertEqual(receipt["signing_run_attempt"], 1)


if __name__ == "__main__":
    unittest.main()
