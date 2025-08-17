#!/usr/bin/env bash
set -e
python src/eval/perplexity.py --config configs/small_125M.yaml --ckpt_dir checkpoints/small_125M
