"""ReVert audit: causal faithfulness auditing of forensic MLLM explanations
via pixel-exact reversion.

Pipeline (all stages are resumable and communicate via JSONL files):
    samples.jsonl --stage1--> queries --detector--> answers
                  --stage2--> intervention queries --detector--> answers
                  --analyze--> metrics table
"""

__version__ = "0.13.6"
