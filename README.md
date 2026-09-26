# Logo Tampering Detection

Research exercise: given ~50 phone photos of NORVIA-branded packaging and
3 clean reference images of the genuine logo, decide which photos show a
subtly tampered logo (stretched, recoloured, re-spaced, or a swapped
letterform) — no labels provided.

**Start here: [`WRITEUP.md`](./WRITEUP.md)** — problem framing, approaches
considered, the experiment actually run, results, next steps, and the AI
tool log.

## Repository layout

- **[`code/`](./code)** — all project code. This is the full
  implementation: the detection pipeline, the synthetic calibration
  benchmark, and the script that produced every number in the write-up.
  Runs end-to-end with a single command — see [`code/README.md`](./code/README.md)
  for exact instructions and a breakdown of what each module does.
- **`candidate_package/`** — the provided dataset, unmodified:
  - `images/` — the 50 candidate photos (`001.jpg`–`050.jpg`)
  - `reference/` — the 3 clean reference logo images
  - `README.txt` — the dataset's own original description
- **`WRITEUP.md`** — the write-up deliverable.

## Quickest way to reproduce the results

```bash
cd code
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python run.py
```

Outputs land in `code/outputs/` (`real_scores.csv`,
`synthetic_calibration.csv`, debug crops) — the same files the numbers in
`WRITEUP.md` are drawn from.
