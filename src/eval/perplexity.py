# src/eval/perplexity.py
import argparse, json, math
from pathlib import Path
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM
import torch
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=False)  # kept for symmetry; not strictly used
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--data_dir", default="data/packed")
    ap.add_argument("--max_eval", type=int, default=200, help="cap on number of sequences to eval")
    args = ap.parse_args()

    ckpt = Path(args.ckpt_dir)
    tok_path = ckpt / "tokenizer.json"
    if not tok_path.exists():
        # fall back to tokenizer shipped with checkpoint dir if present;
        # otherwise try artifacts/tokenizer
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

    if not Path(args.data_dir).exists():
        raise FileNotFoundError(f"Packed dataset not found at {args.data_dir}. Run prepare_corpus.py first.")

    ds = load_from_disk(args.data_dir)
    n = min(len(ds), args.max_eval)
    if n == 0:
        raise RuntimeError("Dataset is empty; add more text or reduce block_size in your config.")
    ds = ds.select(range(n))

    losses = []
    with torch.no_grad():
        for ex in tqdm(ds, desc="Eval PPL"):
            ids = torch.tensor([ex["input_ids"]], dtype=torch.long, device=device)
            outputs = model(input_ids=ids, labels=ids)
            losses.append(float(outputs.loss))

    mean_loss = sum(losses) / len(losses)
    ppl = math.exp(mean_loss)

    out = {"mean_loss": mean_loss, "perplexity": ppl, "n": n}
    out_path = Path(args.ckpt_dir, "eval_perplexity.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[SUCCESS] Perplexity: {ppl:.3f} on n={n} sequences -> {out_path}")

if __name__ == "__main__":
    main()
