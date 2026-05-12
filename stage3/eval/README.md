# Stage-3 Evaluation

`run_eval.py` provides a single evaluation entrypoint for the public Stage-3 release.

It expects:

- a saved Stage-3 model directory
- an evaluation JSON file
- a video root directory or a JSON mapping of multiple video roots

The script is intended to replace the private per-device and ad hoc evaluation variants from the original research workspace.
