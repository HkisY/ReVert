# Sample manifest

`single.csv` and `two_object.csv` list every sample produced by the three inpainting models, including the samples
rejected by the visual inspection. `single_final.jsonl.gz` and `two_object_final.jsonl.gz` are the 1,200 + 1,200
samples that were audited, in the order used, with the edit masks as COCO RLE (`mask_rle`; two-object samples also
carry `parts[i].mask_rle`).

| Column | Meaning |
|---|---|
| `sample_id` | `coco<image id>[p]_<inpainting model>`; `p` marks two-object samples. Same id as in `data/outputs/`. |
| `set` | `single` or `two_object` |
| `inpainter` | `sdxl`, `flux_fill` or `kandinsky22` |
| `build`, `batch` | build folder; `extension` marks the 143 Kandinsky two-object samples generated after the first inspection |
| `coco_image_id`, `coco_ann_id` (`_A`, `_B`) | COCO val2017 photograph and edited instance(s) |
| `source_category` (`_A`, `_B`) | COCO category of the edited object |
| `edit_type` (`_A`, `_B`) | `replace` or `remove` |
| `target_category` (`_A`, `_B`) | category inserted by a replacement (empty for removals) |
| `prompt` (`_A`, `_B`) | prompt given to the inpainting model |
| `seed` (`seed_A`, `seed_B`) | seed of the inpainting model for this sample (object) |
| `width`, `height` | size after the centre crop to multiples of 64 (no resizing) |
| `mask_dilate_px` | dilation of the COCO instance mask (7 px) |
| `mask_frac` | fraction of the image covered by the edit mask |
| `change_in_mask` | mean absolute change inside the mask (grey levels) |
| `qc_auto_pass`, `qc_auto_reason` | automatic check (mask area 1–40 %, mean change ≥ 4 grey levels) |
| `qc_manual_keep`, `qc_manual_note` | visual inspection: `False` = rejected as a failed edit |
| `in_final_set` | part of the 400 samples per model kept by the seeded selection (seed 0) |

Seeds: the photographs are visited in an order fixed by seed 2 (single-object set) or 3 (two-object set), and the
per-sample seeds are drawn from the same generator; the three models receive the same prompt and seed for the
same photograph.
