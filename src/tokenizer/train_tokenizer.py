import argparse, os
from pathlib import Path
import yaml
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import TemplateProcessing

def iter_files(text_dir):
    for p in Path(text_dir).rglob("*.txt"):
        yield str(p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--text_dir", required=True)
    ap.add_argument("--out_dir", default="artifacts/tokenizer")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    vocab_size = int(cfg.get("vocab_size", 32000))
    special_tokens = cfg.get("special_tokens", ["<|pad|>", "<s>", "</s>", "<|unk|>"])

    os.makedirs(args.out_dir, exist_ok=True)

    # Byte-level BPE with add_prefix_space=True (matches GPT-2-ish behavior)
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=special_tokens
    )

    files = list(iter_files(args.text_dir))
    if not files:
        Path(args.text_dir).mkdir(parents=True, exist_ok=True)
        seed = Path(args.text_dir) / "seed.txt"
        seed.write_text(
            "Hello world. This tiny seed exists to train a demo tokenizer.\n",
            encoding="utf-8"
        )
        files = [str(seed)]

    tokenizer.train(files, trainer)

    # Add BOS/EOS post-processing and proper byte-level decoder for clean spaces
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )
    tokenizer.decoder = ByteLevelDecoder()

    out_file = Path(args.out_dir) / "tokenizer.json"
    tokenizer.save(str(out_file))
    print(f"[SUCCESS] Saved tokenizer to {out_file}")

if __name__ == "__main__":
    main()
