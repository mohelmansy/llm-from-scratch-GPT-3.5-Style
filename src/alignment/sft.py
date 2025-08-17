
# Minimal SFT hook (placeholder): shows how to fine-tune on instruction data in JSONL format:
# {"instruction": "...", "input": "...", "output": "..."}
# For a portfolio demo, you can run a few hundred examples to showcase alignment.

import json, argparse
from pathlib import Path
from datasets import Dataset
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out_dir", default="checkpoints/sft_demo")
    args = ap.parse_args()

    data = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    def format_ex(row):
        prompt = f"### Instruction:\n{row['instruction']}\n\n### Input:\n{row.get('input','')}\n\n### Response:\n{row['output']}"
        return {"text": prompt}

    ds = Dataset.from_list([format_ex(r) for r in data])
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(Path(args.ckpt_dir, "tokenizer.json")))
    tokenizer.pad_token = "<|pad|>"
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"

    def tok(b):
        out = tokenizer(b["text"])
        out["labels"] = out["input_ids"].copy()
        return out

    ds = ds.map(tok, batched=True, remove_columns=["text"])
    model = LlamaForCausalLM.from_pretrained(args.ckpt_dir)

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    targs = TrainingArguments(output_dir=args.out_dir, per_device_train_batch_size=2, num_train_epochs=1, report_to=["none"])
    tr = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator, tokenizer=tokenizer)
    tr.train()
    tr.save_model(args.out_dir)
    Path(args.out_dir, "tokenizer.json").write_text(Path(args.ckpt_dir, "tokenizer.json").read_text())

if __name__ == "__main__":
    main()
