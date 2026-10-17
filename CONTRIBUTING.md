# Contributing

Use GitHub issues to propose changes and pull requests to submit them. Describe the expected behavior, the observed behavior and a minimal reproducible example. Run `python -m unittest discover -s tests -v` before submitting a Python change.

## Acceptance requirements

Use descriptive names, explicit imports, type annotations for public interfaces and documented input/output contracts. Python code must pass the project's Ruff rules (`E4`, `E7`, `E9`, `F` and `B`) and all Bandit security checks. Fix findings before merging; discuss any proposed exception with a concrete explanation and a regression check. Preserve source attribution and the project's per-file licenses.

Every major functionality change must include automated tests of its expected behavior and relevant failure cases. Bug fixes should include a regression test. Add tests in the same pull request as the change. Update interface documentation and release notes when behavior changes. Tests use synthetic or explicitly shareable inputs, and must remain CPU-runnable.

Install the development extras in your project environment, then run:

```sh
python -m pip install -e '.[dev]'
python -m ruff check --no-cache apr_repair tests
python -m bandit -r apr_repair
python -m coverage run -m unittest discover -s tests -v
python -m coverage report
python -m build --wheel
python -m pip_audit --local --skip-editable
python -m reuse lint
```

GitHub Actions runs these checks on pushes and pull requests. Coverage measures engineering test execution; scientific performance remains tied to the archived experiment records. The release procedure and security design are documented in [`docs/SECURE_DEVELOPMENT.md`](docs/SECURE_DEVELOPMENT.md).

Original inference modules are mapped to their frozen source hashes. Changes to geometry, preprocessing, weights or numerical behavior should be clearly versioned and accompanied by an explicit comparison to the released implementation. Reported research aggregates retain their published population and measurement boundary.

Contributions to code are accepted under MIT; original documentation contributions use CC BY 4.0. Keep the existing per-file licenses and attribution when working with example assets.
