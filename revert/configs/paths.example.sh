# Copy to configs/paths.sh and fill in your paths:  cp configs/paths.example.sh configs/paths.sh
# scripts/run_audit.sh sources configs/paths.sh. Nothing here is needed to recompute the paper's numbers
# from the released outputs (scripts/reproduce_tables.py).

# Python interpreters of the four environments in environment/
export PY_AUDIT="/path/to/envs/audit/bin/python"     # data generation, audit pipeline, analysis
export PY_SIDA="/path/to/envs/sida/bin/python"       # SIDA-7B worker
export PY_FSDTE="/path/to/envs/fsdte/bin/python"     # FakeShield DTE-FDM worker (transformers 4.37.2)
export PY_FSMFLM="/path/to/envs/fsmflm/bin/python"   # FakeShield MFLM worker (transformers 4.28.0)

# Third-party code (cloned as described in README.md, "Third-party models") and weights
export SIDA_REPO="/path/to/SIDA"
export SIDA_WEIGHTS="/path/to/SIDA-7B"
export FS_REPO="/path/to/FakeShield"
export FS_WEIGHTS="/path/to/fakeshield-v1-22b"      # contains DTE-FDM/ and MFLM/

# COCO 2017 (python -m revert_audit.data.download --root "$COCO_ROOT" --parts val annotations)
export COCO_ROOT="/path/to/coco"

# Hugging Face cache for the inpainting models and CLIP (optional; default ~/.cache/huggingface)
# export HF_HOME="/path/to/hf_home"

# GPU memory budget in GiB (weights beyond it are offloaded to CPU RAM; results are unchanged)
export SIDA_GPU_MEM="${SIDA_GPU_MEM:-22}"
export FS_GPU_MEM="${FS_GPU_MEM:-18}"
export FS_MFLM_GPU_MEM="${FS_MFLM_GPU_MEM:-0}"
