#!/usr/bin/env bash
# Commands behind every number of the paper. Needs configs/paths.sh (copy configs/paths.example.sh).
#
#   bash scripts/run_audit.sh sida                 # single-object set, SIDA-7B (partial reversion)
#   bash scripts/run_audit.sh fakeshield           # single-object set, FakeShield
#   bash scripts/run_audit.sh component-sida       # two-object set, component reversion (after "sida")
#   bash scripts/run_audit.sh component-fakeshield # two-object set, component reversion (after "fakeshield")
#   bash scripts/run_audit.sh analyze sida         # diagnose, boundary attraction, H-COV / H-OOD (same for fakeshield)
#   bash scripts/run_audit.sh tables               # recompute the paper's numbers from data/outputs (CPU only)
#
# SINGLE / PAIRS: samples.jsonl of the regenerated forgeries (scripts/generate_from_manifest.py writes one per
# editor; concatenate them, e.g. cat regen/single_*/samples.jsonl > regen/single.jsonl).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[ -f configs/paths.sh ] && source configs/paths.sh
SINGLE="${SINGLE:-regen/single.jsonl}"; PAIRS="${PAIRS:-regen/two_object.jsonl}"
PY="${PY_AUDIT:-python}"

sida_args() {
    echo --detector sida --sida-python "$PY_SIDA" --sida-repo "$SIDA_REPO" --sida-weights "$SIDA_WEIGHTS" \
        --sida-protocol paper --gpu-mem "$SIDA_GPU_MEM" --batch-size "${SIDA_BATCH:-4}"
}
fs_args() {
    echo --detector fakeshield --fs-dte-python "$PY_FSDTE" --fs-mflm-python "$PY_FSMFLM" --fs-repo "$FS_REPO" \
        --fs-weights "$FS_WEIGHTS" --gpu-mem "$FS_GPU_MEM" --cpu-mem "${FS_CPU_MEM:-40}" \
        --fs-mflm-gpu-mem "$FS_MFLM_GPU_MEM" --batch-size "${FS_BATCH:-16}"
}

case "${1:-}" in
    sida)        # stage 1 (x_f, x_raw, x_r), stage 2 (model / true-edit / random explanations, 3 controls each)
        # shellcheck disable=SC2046
        "$PY" -m revert_audit.cli pilot --samples "$SINGLE" --run runs/full_sida --tag sida7b --steps 1200 \
            --s2-bases fake,raw $(sida_args) --mask-dir runs/full_sida/sida_masks ;;
    fakeshield)
        # shellcheck disable=SC2046
        "$PY" -m revert_audit.cli pilot --samples "$SINGLE" --run runs/full_fakeshield --tag fakeshield --steps 1200 \
            --s2-bases fake $(fs_args) --mask-dir runs/full_fakeshield/masks --fs-cache runs/full_fakeshield/fs_cache.jsonl ;;
    component-sida)
        # shellcheck disable=SC2046
        "$PY" -m revert_audit.cli component --samples "$PAIRS" --run runs/e4b_full_sida --tag sida7b \
            --ref-run runs/full_sida $(sida_args) --mask-dir runs/e4b_full_sida/sida_masks ;;
    component-fakeshield)
        # shellcheck disable=SC2046
        "$PY" -m revert_audit.cli component --samples "$PAIRS" --run runs/e4b_full_fakeshield --tag fakeshield \
            --ref-run runs/full_fakeshield $(fs_args) --mask-dir runs/e4b_full_fakeshield/masks \
            --fs-cache runs/e4b_full_fakeshield/fs_cache.jsonl ;;
    analyze)
        det="${2:?sida|fakeshield}"; tag=sida7b; [ "$det" = fakeshield ] && tag=fakeshield
        R="runs/full_$det"
        "$PY" -m revert_audit.diagnose --run "$R" --tag "$tag" --samples "$SINGLE"
        "$PY" -m revert_audit.attraction --run "$R" --tag "$tag" --samples "$SINGLE"
        "$PY" -m revert_audit.hypotheses --run "$R" --tag "$tag" --samples "$SINGLE" ;;
    tables)
        "$PY" scripts/reproduce_tables.py --data data ;;
    *) sed -n '2,12p' "$0"; exit 2 ;;
esac
