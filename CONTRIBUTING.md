# Contributing

Start with a focused issue describing the observed behavior, expected outcome, Windows/Python versions, and a synthetic reproduction. Never attach real credentials, private interviews, or candidate documents. Use placeholder content and redact paths.

Install the development dependencies and run `python -m pytest tests -q` from the repository root. For capture or inference changes, add a meaningful regression and describe the hardware and exact test boundary. A loopback recording is not a two-end Zoom test. UI changes should include a screenshot using synthetic text.

Preserve independent workers for recognition, translation and answering; complete visible text; user-controlled capture; explicit cloud consent; and isolated local history. Tests must use temporary data directories and must not call paid services. Keep runtime data out of source control. Run `python scripts/check_public_tree.py` before submitting.

Pull requests should explain the concrete behavior change, validation, and material limitations. Keep licensing notices for dependencies or model adapters. Application contributions are provided under the repository's MIT license.

