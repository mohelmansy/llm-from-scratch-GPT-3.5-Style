# src/eval/generation_eval.py
import argparse
from pathlib import Path
import re
import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM, GenerationConfig

def _clean_text(txt: str) -> str:
    # 1) Replace the special "Ġ" marker with a normal space
    txt = txt.replace("Ġ", " ")
    # 2) If the whole string is "over-spaced" like "t h i s", collapse those
    # Detect long runs of single letters separated by spaces
    def deletterize(match: re.Match) -> str:
        s = match.group(0)
        return s.replace(" ", "")
    # Collapse runs of "a b c d" of length >= 3 letters
    txt = re.sub(r"(?:\b[A-Za-z]\b(?:\s+\b[A-Za-z]\b){2,})", deletterize, txt)
    # Normalize multiple spaces
    txt = re.sub(r"\s{2,}", " ", txt).strip()
    return txt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--prompt", default="Write a haiku about transformers.")
    args = ap.parse_args()

    ckpt = Path(args.ckpt_dir)
    tok_path = ckpt / "tokenizer.json"
    if not tok_path.exists():
        alt = Path("artifacts/tokenizer/tokenizer.json")
        tok_path = alt if alt.exists() else tok_path
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {tok_path} or artifacts/tokenizer/tokenizer.json")

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tok_path))
    tokenizer.pad_token = "<|pad|>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LlamaForCausalLM.from_pretrained(args.ckpt_dir).to(device)
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt")
    inputs.pop("token_type_ids", None)          # LLaMA doesn't use these
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_cfg = GenerationConfig(
        max_new_tokens=120,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        out = model.generate(**inputs, generation_config=gen_cfg)

    raw_text = tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    text = _clean_text(raw_text)

    out_path = ckpt / "sample_generation.txt"
    out_path.write_text(text, encoding="utf-8")
    print("----- Generated text -----")
    print(text)
    print(f"\n[SAVED] {out_path}")

if __name__ == "__main__":
    main()
