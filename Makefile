
.PHONY: setup tokenizer prepare train-small eval-small

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt || true

tokenizer:
	python src/tokenizer/train_tokenizer.py --config configs/tokenizer.yaml --text_dir data/sample_text

prepare:
	python src/data/prepare_corpus.py --config configs/small_125M.yaml --text_dir data/sample_text --out_dir data/packed

train-small:
	python src/train.py --config configs/small_125M.yaml

eval-small:
	python src/eval/perplexity.py --config configs/small_125M.yaml --ckpt_dir checkpoints/small_125M
