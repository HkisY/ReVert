# Pointing Is Not Proof: Auditing the Self-Explanations of Forensic MLLMs

Code, sample manifest and model outputs for **ReVert**, a framework that audits the masks forensic multimodal
large language models (MLLMs) give as evidence for their verdicts. ReVert restores the region a model points to
to its original pixels and observes whether the verdict and the pointed-at region change.

> Anonymous submission, under review. Please do not distribute.

## Overview

For a local forgery the original photograph is known. Restoring a region `A` of the forgery `x_f` to the
original `x_r`,

```
R(x_f, A) = A ⊙ x_r + (1 − A) ⊙ x_f ,
```

removes the evidence inside `A` without introducing any new pixel value. ReVert uses this pixel-exact reversion
to measure, for each explanation, *necessity* (how much the fake score drops; CNG and the faithfulness ratio R),
*persistence* (whether the model still flags the image and keeps pointing at the restored region; PER, and cPER
under component reversion) and *specificity* (whether it points at the forged pixels or merely at an object;
MNG, cMNG, switch rate). Every effect is calibrated between two references: restoring the true edit and restoring
a random region of the same shape. Component reversion restores one of two independently edited objects as a
whole, so that no new boundary between authentic and forged pixels is created.

## Repository structure

```
README.md  LICENSE  pyproject.toml  requirements.txt
environment/            conda environments used for the paper (versions pinned): audit, sida, fsdte, fsmflm
configs/
  gates.json            thresholds of the decision gates and stop rules used by every run
  paths.example.sh      paths to interpreters, third-party code, weights and COCO (copy to configs/paths.sh)
revert_audit/           the library
  interventions.py      reversion operator R, keep operator, softened boundaries
  regions.py            masks, overlaps, shape-matched controls (rigid translations), in-edit controls
  pipeline.py           stage 1 (original images) and stage 2 (reverted images and controls); CNG, PER, MNG, DR
  component.py          component reversion on two-object forgeries (cPER, cSTAY, switch, cMNG)
  attraction.py         boundary attraction BA        seam.py   softened-boundary reruns
  diagnose.py           R with clean controls, coverage bins, strict PER, STAY − PER, per-model tables
  hypotheses.py         H-COV and H-OOD (bootstrap clustered on source photographs)
  rationale.py          persistence of FakeShield's textual rationale (exploratory)
  metrics.py            measures and bootstrap         gates.py / pilot.py   decision gates, progressive runs
  data/                 forgery construction (sources.py, editors.py, build.py, qc.py, merge.py)
  detectors/            SIDA-7B, FakeShield, So-Fake-R1 and zero-shot Qwen2.5-VL wrappers (no third-party code)
scripts/
  reproduce_tables.py   recompute the paper's numbers from data/outputs (CPU only)
  supp_numbers.py       photo-clustered intervals, matched baselines, false-positive rates (needs run folders)
  generate_from_manifest.py   regenerate the forgeries from the manifest
  run_audit.sh          the commands behind every run
  make_teaser.py  make_fig3.py  make_fig4.py  export_fig2_assets.py   figures
  export_release_data.py      how data/ was exported from the run folders
tests/                  unit and integration tests with mock detectors (CPU)
data/
  manifest/             every generated sample, including rejected ones, and the final evaluation sets
  outputs/              every model answer (verdict, fake score, mask as COCO RLE, rationale)
```

## Installation

The analysis of the released outputs needs only the CPU environment:

```bash
pip install numpy pillow opencv-python-headless pycocotools matplotlib
python scripts/reproduce_tables.py --data data
```

Running the detectors or regenerating forgeries needs the four conda environments in `environment/`
(the detectors depend on incompatible `transformers` versions, so each runs in its own process and
environment):

```bash
conda env create -f environment/audit.yml    # data generation, audit pipeline, analysis
conda env create -f environment/sida.yml     # SIDA-7B
conda env create -f environment/fsdte.yml    # FakeShield DTE-FDM (transformers 4.37.2)
conda env create -f environment/fsmflm.yml   # FakeShield MFLM (transformers 4.28.0)
cp configs/paths.example.sh configs/paths.sh # then edit the paths
python scripts/check_env.py                  # sanity check of the environment
```

Hardware: a GPU with at least 24 GB of memory; FakeShield's DTE-FDM (fp16) additionally needs about 40 GB of
CPU RAM for the weights that do not fit on the GPU (offloading moves weights only and does not change results).

## Third-party models

No weights or third-party code are included. Each model is used under its own license.

| Model | Code | Weights | Used for |
|---|---|---|---|
| SIDA-7B | official SIDA repository (linked from the model card), main branch, October 2026 | [`saberzl/SIDA-7B`](https://huggingface.co/saberzl/SIDA-7B) | audited detector |
| FakeShield v1-22b | [`zhipeixu/FakeShield`](https://github.com/zhipeixu/FakeShield), main branch, October 2026 | [`zhipeixu/fakeshield-v1-22b`](https://huggingface.co/zhipeixu/fakeshield-v1-22b) | audited detector |
| So-Fake-R1 | official release (inference v5.64) | released LoRA adapters on Qwen2.5-VL-7B-Instruct | tested, not audited (Supp. A) |
| Qwen2.5-VL-7B-Instruct | – | `Qwen/Qwen2.5-VL-7B-Instruct` | zero-shot reference (Supp. A) |
| SDXL Inpainting 0.1 | diffusers | `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | inpainting model |
| FLUX.1-Fill [dev] | diffusers | `black-forest-labs/FLUX.1-Fill-dev` | inpainting model |
| Kandinsky 2.2 | diffusers | `kandinsky-community/kandinsky-2-2-decoder-inpaint` + `kandinsky-community/kandinsky-2-2-prior` | inpainting model |

Both audited detectors run with their official code, weights and prompts under greedy decoding. The worker
processes in `revert_audit/detectors/` import the official code from the cloned repositories; the fake-score
readout is defined in one place per detector:

* SIDA: `s = 1 − p(real)` from the classification token (`detectors/sida_worker.py`);
* FakeShield: `s = p(y⁺) / (p(y⁺) + p(y⁻))` after the fixed answer prefix
  `"1. Whether the picture has been tampered with / Description of the tampered area: The picture has"`,
  with `y⁺ = "been"` and `y⁻ = "not"` (`detectors/fakeshield_worker.py`, constants `PREFIX`, `NOT_TAMPERED`);
  the verdict follows FakeShield's own rule (an answer containing "has not been tampered with" is real).
  The official release samples with temperature 0.2; we decode greedily;
* zero-shot Qwen2.5-VL: prompt `PROMPT` and forced prefix `SCORE_PREFIX` in `detectors/qwen_vl.py`.

## Data

**COCO.** `python -m revert_audit.data.download --root /path/to/coco --parts val annotations` downloads COCO
val2017 and its instance annotations.

**Manifest.** `data/manifest/` lists every generated sample of the single-object and two-object sets, including
those rejected by the visual inspection, so that the selection of the 1,200 + 1,200 evaluation samples can be
followed exactly (each inpainting model produced 440 samples; after inspection 400 per model were kept by a
seeded random selection). Column definitions: [`data/manifest/README.md`](data/manifest/README.md).

**Regenerating the forgeries.** Forgeries and original photographs are not redistributed. They are regenerated
from the manifest (same photograph, crop, edit mask, prompt and seed):

```bash
python scripts/generate_from_manifest.py --set single --coco-root /path/to/coco --editor sdxl --out regen/single_sdxl
```

Generated pixels can differ slightly across GPUs and library versions; the versions used are in
`environment/audit.yml`. Inpainting settings: SDXL Inpainting 0.1 fp16, 30 steps, guidance 8.0, strength 0.99;
FLUX.1-Fill [dev] bf16, 50 steps, guidance 30; Kandinsky 2.2 fp16, 50 decoder and 25 prior steps, guidance 4.0;
all at the photograph's own resolution (`revert_audit/data/editors.py`).

## Running the audit

```bash
bash scripts/run_audit.sh sida                  # single-object set, SIDA-7B
bash scripts/run_audit.sh fakeshield            # single-object set, FakeShield
bash scripts/run_audit.sh component-sida        # two-object set, component reversion
bash scripts/run_audit.sh component-fakeshield
bash scripts/run_audit.sh analyze sida          # R with clean controls, BA, H-COV / H-OOD (also: fakeshield)
```

The SIDA run also queries the full inpainter outputs `x_raw` (robustness analysis, Supp. E.5). The
softened-boundary experiment (Supp. B, pilot data) is `python -m revert_audit.cli seam --run <run> --tag sida7b
--ramps 8,24 ...`; So-Fake-R1 and zero-shot Qwen2.5-VL (Supp. A) are run with `python -m revert_audit.cli pilot
--detector sofake ...` and `--detector qwen ...`.

## Reproducing the paper's numbers from the released outputs

```bash
python scripts/reproduce_tables.py --data data           # no GPU, no weights, no images
```

The script expands `data/outputs/` (masks as COCO RLE) into the folder layout the experiment code writes, runs
the unchanged analysis code on it, and compares every number with the value printed in the paper. Add
`--keep work` to keep the expanded folders; the figure and supplementary scripts can then be pointed at them.

| Paper | Script | Input |
|---|---|---|
| Tab. 1, Supp. Tab. E1 | `scripts/reproduce_tables.py`, `scripts/supp_numbers.py` (photo-clustered intervals, false-positive rates) | `data/outputs/single_*`, `component_*` |
| Fig. 1 (c), Supp. Tab. E4–E5 | `scripts/reproduce_tables.py` (component reversion) | `data/outputs/component_*` |
| Fig. 3, Supp. Tab. E2 | `scripts/make_fig3.py --layout column --sida-run work/outputs/single_sida --fs-run work/outputs/single_fakeshield --samples work/manifest/single_final.jsonl` | expanded outputs |
| Supp. Tab. E3 (H-COV, H-OOD) | `scripts/reproduce_tables.py` (`revert_audit.hypotheses`) | `data/outputs/single_*` |
| Supp. Tab. B1 (boundary attraction) | `scripts/reproduce_tables.py` (`revert_audit.attraction`) | `data/outputs/single_*` |
| Supp. Tab. C1 (evaluation sets) | `data/manifest/*.csv` | manifest |
| Supp. Tab. E6 (rationales) | `python -m revert_audit.rationale --run work/outputs/single_fakeshield ...` | expanded outputs, COCO annotations |
| Supp. Tab. A1 | pilot runs (`python -m revert_audit.cli pilot --detector sofake` / `qwen`) | pilot data |
| Fig. 1, Fig. 4, Supp. Fig. F1 | `scripts/make_teaser.py`, `scripts/make_fig4.py` (`--random 4 --seed 0` for Fig. F1) | regenerated images and run folders |

Bootstrap settings: 1,000 resamples for the main measures and 2,000 for hypothesis tests, comparisons across
inpainting models and Fig. 3; seeds are fixed in the code. Controls: each explanation has 3 shape-matched
controls and up to 3 in-edit controls, with seeds derived from the sample identifier
(`revert_audit/pipeline.py`, `stage2`).

## License

The code is released under the MIT License (see `LICENSE`). COCO images and annotations, the inpainting models
and the audited detectors are subject to their own licenses and terms of use.

## Citation

Anonymous. Under review.
