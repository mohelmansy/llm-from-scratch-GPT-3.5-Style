# src/train.py
import argparse
import json
import os
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from transformers import (
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.modeling.build_model import build_llama_like, ModelSpec


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _maybe_inherit(cfg: dict) -> dict:
    """
    Lightweight 'inherit' support:
    If cfg contains 'inherit': "configs/base.yaml", load base then shallow-update with cfg.
    """
    if "inherit" in cfg and cfg["inherit"]:
        base_path = Path(cfg["inherit"])
        base = _load_yaml(str(base_path))
        # prevent recursive loops; apply parent first, then override with child
        child = dict(cfg)
        child.pop("inherit", None)
        merged = {**base, **child}
        # also merge nested sections commonly used here
        for key in ("model", "train", "special_tokens"):
            if key in base or key in child:
                merged[key] = {**base.get(key, {}), **child.get(key, {})}
        return merged
    return cfg


def _bool(v, default=False):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes", "y", "t")
    return bool(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config (base.yaml or small_125M.yaml)")
    args = ap.parse_args()

    # --- Load config (with optional inheritance) ---
    cfg_raw = _load_yaml(args.config)
    cfg = _maybe_inherit(cfg_raw)

    # --- Resolve common fields with safe defaults ---
    tokenizer_dir = cfg.get("tokenizer_dir", "artifacts/tokenizer")
    vocab_size = int(cfg.get("vocab_size", 50257))

    specials = cfg.get("special_tokens", {
        "pad_token": "<|pad|>",
        "bos_token": "<s>",
        "eos_token": "</s>",
    })

    mcfg = cfg.get("model", {})
    tspec = ModelSpec(
        hidden_size=int(mcfg.get("hidden_size", 768)),
        n_layer=int(mcfg.get("n_layer", 12)),
        n_head=int(mcfg.get("n_head", 12)),
        intermediate_size=int(mcfg.get("intermediate_size", 2048)),
        max_position_embeddings=int(mcfg.get("max_position_embeddings", 1024)),
        vocab_size=int(vocab_size),
        rope_theta=float(mcfg.get("rope_theta", 100000.0)),
    )

    tcfg = cfg.get("train", {})
    output_dir = tcfg.get("output_dir", "checkpoints/small_125M")
    eval_every = int(tcfg.get("eval_every", 200))
    save_every = int(tcfg.get("save_every", 200))

    per_device_train_batch_size = int(tcfg.get("per_device_train_batch_size", 2))
    per_device_eval_batch_size = int(tcfg.get("per_device_eval_batch_size", 2))
    gradient_accumulation_steps = int(tcfg.get("gradient_accumulation_steps", 8))
    learning_rate = float(tcfg.get("learning_rate", 3.0e-4))
    weight_decay = float(tcfg.get("weight_decay", 0.1))
    warmup_ratio = float(tcfg.get("warmup_ratio", 0.05))
    logging_steps = int(tcfg.get("logging_steps", 20))
    lr_scheduler_type = tcfg.get("lr_scheduler_type", "cosine")
    # precision flags (bf16 if hw supports; else fp16 on CUDA; else cpu fp32)
    want_bf16 = _bool(tcfg.get("bf16", False), default=False)
    use_cuda = torch.cuda.is_available()
    can_bf16 = use_cuda and torch.cuda.is_bf16_supported()
    bf16_flag = bool(want_bf16 and can_bf16)
    fp16_flag = bool(use_cuda and not bf16_flag)

    # --- Make sure tokenizer exists ---
    tok_file = Path(tokenizer_dir) / "tokenizer.json"
    if not tok_file.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at '{tok_file}'. "
            f"Run tokenizer training first:\n"
            f"  python src/tokenizer/train_tokenizer.py --config configs/tokenizer.yaml --text_dir data/sample_text --out_dir {tokenizer_dir}"
        )

    # --- Build model ---
    model = build_llama_like(tspec)

    # --- Load tokenizer ---
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tok_file))
    tokenizer.pad_token = specials.get("pad_token", "<|pad|>")
    tokenizer.bos_token = specials.get("bos_token", "<s>")
    tokenizer.eos_token = specials.get("eos_token", "</s>")

    # --- Load dataset (expects 'data/packed' created by prepare_corpus.py) ---
    data_dir = "data/packed"
    if not Path(data_dir).exists():
        raise FileNotFoundError(
            f"Tokenized dataset not found at '{data_dir}'. "
            f"Create it via:\n"
            f"  python src/data/prepare_corpus.py --config {args.config} --text_dir data/sample_text --out_dir data/packed"
        )
    ds = load_from_disk(data_dir)

    # Map into labels = input_ids (teacher forcing)
    def fmt(examples):
        return {"input_ids": examples["input_ids"], "labels": examples["input_ids"]}
    ds = ds.map(fmt, batched=True, remove_columns=[])

    # For a quick demo run, keep a small subset so it runs fast
    train_sel = min(2000, len(ds))
    eval_sel = min(200, len(ds))
    train_ds = ds.shuffle(seed=42).select(range(train_sel))
    eval_ds = ds.select(range(eval_sel))

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # --- TrainingArguments (v4/v5 compatible) ---
    os.makedirs(output_dir, exist_ok=True)
    common_kwargs = dict(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_total_limit=3,
        lr_scheduler_type=lr_scheduler_type,
        report_to=[],  # [] works for v5; avoids unknown "none" for some installs
        num_train_epochs=1,  # demo epoch; adapt as needed
        bf16=bf16_flag,
        fp16=fp16_flag,
    )

    try:
        # Transformers v4.x keyword
        targs = TrainingArguments(
            **common_kwargs,
            evaluation_strategy="steps",
            eval_steps=eval_every,
            save_steps=save_every,
        )
    except TypeError:
        # Transformers v5.x renamed 'evaluation_strategy' -> 'eval_strategy'
        targs = TrainingArguments(
            **common_kwargs,
            eval_strategy="steps",
            eval_steps=eval_every,
            save_steps=save_every,
        )

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # --- Train ---
    trainer.train()

    # --- Save model + a tiny training summary ---
    trainer.save_model(output_dir)
    # Save the tokenizer alongside the checkpoint for convenience
    Path(output_dir, "tokenizer.json").write_text(Path(tok_file).read_text(encoding="utf-8"), encoding="utf-8")

    # Save last logs/tail metrics if available
    state = getattr(trainer, "state", None)
    if state and getattr(state, "log_history", None):
        tail = state.log_history[-10:]
        with open(Path(output_dir, "metrics_tail.json"), "w", encoding="utf-8") as f:
            json.dump(tail, f, indent=2)

    print(f"[SUCCESS] Training complete. Checkpoint saved to: {output_dir}")


if __name__ == "__main__":
    main()
