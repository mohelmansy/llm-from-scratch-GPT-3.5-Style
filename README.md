# 🚀 LLM-from-Scratch: Training & Evaluating a Transformer Language Model  

![Python](https://img.shields.io/badge/Python-3.10-blue.svg) 
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg) 
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg) 
![Status](https://img.shields.io/badge/Status-Demo%20Complete-green.svg)

---

## 📖 Overview
This project is a **showcase** of building a **transformer-based language model (LLM)** entirely **from scratch** and training it on a **single 6GB GPU**.  

It highlights:  
- 🏗️ **Model implementation** (decoder-only Transformer, GPT-style).  
- 🎛️ **Training & checkpointing** with PyTorch.  
- 📊 **Evaluation** using perplexity and text generation.  
- 💡 **Optimizations** for limited compute resources.  

> ⚡ The focus is on the **engineering pipeline**, not achieving SOTA performance — making it an excellent learning and demonstration project.  

---

## 🎯 Key Objectives
✔️ Develop a **scalable and reproducible** LLM pipeline.  
✔️ Train a **100M parameter transformer** under strict VRAM constraints.  
✔️ Showcase **perplexity evaluation** and **text generation**.   

---

## ⚙️ Tech Stack
- **Language**: Python 3.10  
- **Frameworks**: PyTorch, Hugging Face Transformers  
- **Hardware**: Single NVIDIA GPU (6GB VRAM)  
- **Environment**: Virtualenv (`.venv`)  

---

## 📂 Repository Structure

llm-from-scratch/
│── configs/ # Model & training configs
│── data/ # Training datasets
│── checkpoints/ # Model weights & checkpoints
│── results/ # Evaluation outputs (ppl & generations)
│── src/ # Source code
│ ├── train.py # Training script
│ ├── eval/
│ │ ├── perplexity.py # Perplexity evaluation
│ │ └── generation_eval.py# Text generation
│── README.md # Documentation


---

## 🧠 Model Architecture
- **Type**: Decoder-only Transformer (GPT-style)  
- **Parameters**: ~100M  
- **Layers**: 10  
- **Hidden Dim**: 640  
- **Attention Heads**: 10  
- **Context Window**: 192 tokens  
- **Optimizer**: AdamW + cosine decay  
- **Precision**: Mixed FP16 training  

---

🚀 How to Run
1️⃣ Prepare Dataset
python src/data/prepare_corpus.py --config configs/gpu_100M_6GB.yaml --text_dir data/sample_text --out_dir data/packed

2️⃣ Train the Model
python -m src.train --config configs/gpu_100M_6GB.yaml


✔️ Checkpoints are saved to:
checkpoints/gpu_100M_6GB

3️⃣ Evaluate Perplexity
python src/eval/perplexity.py --ckpt_dir checkpoints/gpu_100M_6GB

4️⃣ Generate Text
python src/eval/generation_eval.py --ckpt_dir checkpoints/gpu_100M_6GB --prompt "Explain gas turbines in simple terms."


📂 Generated text is saved in:
checkpoints/gpu_100M_6GB/sample_generation.txt

---

## 📊 Training Process
- **Dataset**: Demo text corpus (toy-scale)  
- **Batch Size**: 32  
- **Steps**: ~800  
- **Training Runtime**: ~23s per step on 6GB GPU  
- **Checkpointing**: automatic save to `checkpoints/`  

**Sample Training Log**:
{'train_runtime': 23.511, 'train_loss': 10.36, 'epoch': 1.0}
[SUCCESS] Training complete → checkpoints/gpu_125M

---

## 📈 Evaluation

### 🔹 Perplexity
```bash
python src/eval/perplexity.py --ckpt_dir checkpoints/gpu_100M_6GB

[SUCCESS] Perplexity: 40097.322 on n=1 sequences

python src/eval/generation_eval.py --ckpt_dir checkpoints/gpu_100M_6GB \
    --prompt "Explain gas turbines in simple terms."

----- Generated text -----
xplain gas turbines in simple terms.

Outputs are stored in:

results/eval_perplexity.json
results/sample_generation.txt

🛠️ Challenges & Solutions
Challenge	Solution
🚧 6GB VRAM limit	Reduced model size, used FP16 training
⚠️ Unused token_type_ids	Removed from generate() kwargs
📉 High perplexity	Acknowledged dataset limitation; positioned as demo
🚀 Future Enhancements

✅ Train on larger open datasets (WikiText, OpenWebText).
✅ Scale beyond 100M parameters using gradient checkpointing.
✅ Add loss & perplexity trend plots.
✅ Introduce fine-tuning (SFT, RLHF) for instruction tasks.
✅ Deploy as an API service with FastAPI/Streamlit.
✅ Key Takeaways

Even with 6GB GPU constraints, a functional LLM pipeline can be implemented.
The project demonstrates core building blocks of modern LLMs.
Serves as both an educational reference and a portfolio artifact for AI engineering roles.

👨‍💻 Author
[Mohamed Elmansy]