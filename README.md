# APR: Compact Analytic-Prior Repair

Research software accompanying **APR: Compact Analytic-Prior Repair for Traffic-Sign Detection**, by **Chih-Yuan Cheng and Bor-Jiunn Hwang**, Ming Chuan University. **IEEE CSIT 2026, Paper 066 — accepted and orally presented on 15 October 2026.** [Official conference program](https://csit.ieee.org.ua/accepted-papers/).

[Archived release DOI](https://doi.org/10.5281/zenodo.23430088) · [Interactive research example](https://huggingface.co/spaces/zzy001zzy/apr-traffic-sign-repair) · [Video demonstration](https://youtu.be/g6abucXnI68) · [Component licenses](#licensing)

APR combines an analytic Telea proposal, a bounded residual refiner and a component selector. Its repair network contains **27,188 parameters**. The selected RGB-only locator operates inside a supplied sign ROI.

## Included in this release

- Original frozen APR-HardSelect-96 inference modules, with source SHA-256 mappings.
- Selected ROI presence-mask cascade architecture and tensor-only Safetensors weights.
- A Python interface for repair with an aligned caller-provided support mask.
- One fixed, attributed DFG qualitative case and the reported aggregate tables.
- CPU engineering checks for checkpoint integrity, tensor contracts and exact outside-mask composition.

The research scope is **source-group-disjoint DFG traffic signs, supplied sign ROIs, and digital non-adaptive occlusion**. The downloadable Python example uses a caller-provided mask. The selected locator's input contract is a normalized float32 `B×3×128×128` sign-ROI tensor.

## Local inference

Use Python 3.12 and a CPU or CUDA-capable PyTorch 2.13 or newer installation:

```sh
git clone https://github.com/zzy001zzy/apr-traffic-sign-repair.git
cd apr-traffic-sign-repair
python -m pip install -e .
python -m unittest discover -s tests -v
```

```python
import cv2
import numpy as np
from apr_repair import load_repair_model, repair_rgb

image = cv2.cvtColor(cv2.imread("input.png"), cv2.COLOR_BGR2RGB)
mask = cv2.imread("support.png", cv2.IMREAD_GRAYSCALE) > 0
model = load_repair_model("weights/apr-hardselect96.safetensors")
repaired, receipt = repair_rgb(model, image, mask.astype(np.bool_))
cv2.imwrite("repaired.png", cv2.cvtColor(repaired, cv2.COLOR_RGB2BGR))
print(receipt)
```

`component_gate_means` describes the repair selector's masked mean gate output. The receipt also records the selected branch and outside-mask byte identity for each call.

## Reported research results

| Quantity | Reported value | Population / boundary |
|---|---:|---|
| Predicted-mask group-mean recovery, Faster R-CNN | 84.3%, descriptive 95% CI 78.8–89.0% | 56 eligible source groups |
| Predicted-mask case-weighted recovery, Faster R-CNN | 569 / 701 = 81.2% | Eligible final attack-caused failures |
| Repair network size | 27,188 parameters | Refiner plus component gate |
| Repair-only mean / p95 latency | 76.717 / 86.117 ms | 128 development cases, 57 groups; batch 1, FP32, RTX 5070 Ti |

The interval resamples whole source groups. Group-weighted and case-weighted recovery are separate estimands. Complete six-method reported results and their manuscript-source hashes are in [`docs/reported-results.json`](docs/reported-results.json).

The web example displays one retained **predicted-mask qualitative comparison** from fixed E14 atlas case `DFGSHAPE_0003465`. These precomputed panels illustrate a single case; the table above summarizes the corresponding reported aggregate study.

## Data and method attribution

The analytic proposal follows Telea's fast-marching inpainting method, with radius 3 specified by the APR protocol. DFG data are attributed to Tabernik and Skočaj and the ViCoS Laboratory, University of Ljubljana; original images were provided and annotated by DFG Consulting d.o.o. [Official dataset and license](https://www.vicos.si/resources/dfg/).

The retained four-panel example was cropped, digitally occluded, repaired and laid out for a noncommercial research comparison. It is shared under **CC BY-NC-SA 4.0**. [`docs/image-attribution.json`](docs/image-attribution.json) records the asset identity and modifications. Verified publication references, including the AOT-GAN and LaMa comparison methods named in the aggregate table, are provided in [`docs/references.txt`](docs/references.txt) and [`docs/references.bib`](docs/references.bib).

## Licensing

| Component | License |
|---|---|
| Project-authored Python, web application and configuration code | MIT |
| Original documentation and reported aggregate tables | CC BY 4.0 |
| Released model weights | CC BY-NC-SA 4.0 |
| DFG-derived qualitative example | CC BY-NC-SA 4.0 |

Full texts are included in `LICENSES/`; file-level assignments are recorded in `REUSE.toml`. Safetensors files preserve the tensor values of the frozen checkpoints. The weight manifest records both the original checkpoint hash and the release-file hash.

## Contributing and security

Please use GitHub issues for reproducible bugs, compatibility reports and documentation suggestions. Include package versions, the command used and a minimal example made from synthetic or shareable inputs. See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`SECURITY.md`](SECURITY.md).

## Artifact verification

Version 1.0.2 is archived at [Zenodo DOI 10.5281/zenodo.23430088](https://doi.org/10.5281/zenodo.23430088). The archive includes the release ZIP, a file-level SPDX inventory and SHA-256 checksums. The published Zenodo files were downloaded and matched byte for byte against the GitHub release assets.

[Artifact checks](https://github.com/zzy001zzy/apr-traffic-sign-repair/actions/workflows/ci.yml) run 12 CPU engineering tests with branch coverage, Ruff code checks, Bandit security analysis, a source-to-wheel build, installed-dependency auditing and REUSE licensing checks. These checks cover software contracts and release engineering; reported scientific results retain their original measurement scope. Development commands, contribution requirements and test policy are in [`CONTRIBUTING.md`](CONTRIBUTING.md); security design and the release procedure are in [`docs/SECURE_DEVELOPMENT.md`](docs/SECURE_DEVELOPMENT.md).

Version 1.0.3 adds this engineering workflow, input-boundary regressions and documented security/reporting processes. Its supported PyTorch minimum is 2.13, addressing upstream advisory CVE-2025-3000; the public CI uses the 2.13.0 CPU wheel. Frozen model bodies, weights, example pixels and research aggregates retain the archived identities. The DOI and `CITATION.cff` identify the separately archived scientific artifact, version 1.0.2. Historical benchmark measurements retain their original runtime and are not remeasured by these engineering checks.

[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/15606/badge)](https://www.bestpractices.dev/projects/15606)

The badge links to the current [OpenSSF Best Practices self-assessment](https://www.bestpractices.dev/en/projects/15606) and its public criterion-level evidence. An [OpenSSF Scorecard v5.6.0 snapshot](docs/scorecard-2026-10-17.json) records 4.2/10 for the initial public release commit; that historical score is separate from the Best Practices badge.

## Qualitative example provenance

The fixed qualitative panels use **predicted support masks**. Version 1.0.0 called this example known-mask; version 1.0.1 corrected the provenance and heading but retained two viewer labels. Version 1.0.2 makes all viewer labels consistent. [Primary producer and asset provenance](docs/qualitative-provenance.json) records the original source/manifest hashes and exact predicted-mask repair calls. The example pixels, inference modules, weights and reported numbers are identical to version 1.0.0. The Python API accepts an aligned caller-provided support mask.
