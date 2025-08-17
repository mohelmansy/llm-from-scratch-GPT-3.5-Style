#!/usr/bin/env bash
set -e
python src/tokenizer/train_tokenizer.py --config configs/tokenizer.yaml --text_dir data/sample_text
python src/data/prepare_corpus.py --config configs/small_125M.yaml --text_dir data/sample_text --out_dir data/packed
python src/train.py --config configs/small_125M.yaml
python src/eval/perplexity.py --config configs/small_125M.yaml --ckpt_dir checkpoints/small_125M
python src/eval/generation_eval.py --ckpt_dir checkpoints/small_125M --prompt "Explain gas turbines in simple terms."
