[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# Deep Learning Training Playbook

> **Practical strategies, engineering patterns, and lessons learned from training deep learning models (2014–2026).**

This repository helps answer one question:

> **"Given my problem, data, compute budget, and deployment constraints, what should I use, when, and why?"**

Instead of summarizing research papers, this playbook transforms them into **engineering decision frameworks** that help you choose the right model, architecture, training strategy, and optimization techniques.

---

# 🚀 Learning Roadmap

The playbook is organized into independent parts. Each part can be read on its own, but together they build a complete understanding of modern deep learning.

| Part | Status | What You'll Learn |
|------|--------|-------------------|
| 📘 **[Part I — Training Strategy](docs/part-1-training-strategy/README.md)** | ✅ Available | Transfer Learning, Foundation Models, LoRA, QLoRA, Continued Pretraining, Knowledge Distillation, Decision Framework |
| 🏗️ **Part II — Model Architectures** | 🚧 Coming Soon | CNNs, ResNet, ConvNeXt, Vision Transformers, CLIP, SigLIP, Mamba, Multimodal Models |
| 🎨 **Part III — Generative Models** | 🚧 Coming Soon | Diffusion Models, Flow Matching, Stable Diffusion, Flux, DreamBooth, Virtual Try-On, Video Generation |
| ⚡ **Part IV — Training Optimization** | 🚧 Coming Soon | AdamW, Lion, Muon, Sophia, Learning Rate Schedulers, Mixed Precision, Distributed Training |
| 📜 **Part V — History of Deep Learning** | 🚧 Coming Soon | AlexNet → Transformers → Foundation Models → Agentic AI |

---

# 📖 Start Here

If you're new to this repository, begin with:

➡ **[Part I — Training Strategy](docs/part-1-training-strategy/README.md)**

You'll learn:

- Why Transfer Learning changed deep learning
- When to use Full Fine-Tuning
- When LoRA is enough
- How QLoRA enables training on consumer GPUs
- When Continued Pretraining outperforms Fine-Tuning
- How Knowledge Distillation reduces inference costs
- How experienced ML engineers choose the right strategy

---

# ⚡ Quick Decision Guide

## Which training strategy should I use?

| Situation | Recommended Strategy |
|-----------|----------------------|
| Small dataset | Transfer Learning |
| Need highest possible accuracy | Full Fine-Tuning |
| Limited GPU memory | LoRA |
| Consumer GPU (RTX 3090/4090) | QLoRA |
| New domain (Medical, Finance, Legal) | Continued Pretraining |
| Expensive inference | Knowledge Distillation |
| Frequently changing knowledge | RAG (instead of training) |

📖 **Complete explanation:**  
**➡️ [Part I — Training Strategy](docs/part-1-training-strategy/README.md)**

---

## Which architecture should I use?

| Problem | Recommended Architecture |
|----------|--------------------------|
| Medical Imaging | ConvNeXtV2 |
| Satellite Images | ConvNeXtV2 |
| Web-scale Vision | SigLIP2 / ViT |
| Long Context | Mamba2 |
| Vision + Language | Qwen2-VL |

*(Covered in Part II.)*

---

## Which generative model should I use?

| Task | Recommended Model |
|------|-------------------|
| Product Images | Flux |
| Posters | Flux |
| Character LoRA | Flux LoRA |
| Virtual Try-On | CatVTON |
| Video Generation | Kling / Runway Gen3 |

*(Covered in Part III.)*

---

## Which optimizer should I use?

| Situation | Recommended Optimizer |
|-----------|-----------------------|
| General Training | AdamW |
| Memory-Constrained Training | Lion |
| Large Model Training | Muon |
| LLM Fine-Tuning | 8-bit AdamW |

*(Covered in Part IV.)*

---

# 🕰 Evolution of Deep Learning

| Year | Breakthrough | Why It Mattered |
|------|--------------|-----------------|
| **2012** | AlexNet | Deep CNNs surpassed traditional computer vision methods |
| **2014** | Transfer Learning | Reuse pretrained representations instead of training from scratch |
| **2017** | Transformer | Introduced self-attention and transformed sequence modeling |
| **2020** | GPT-3 & CLIP | Foundation models became practical |
| **2023** | QLoRA | Fine-tune large language models on consumer GPUs |
| **2024** | Flux | Flow Matching became a practical alternative to diffusion models |
| **2026** | Mamba2, Muon | Faster architectures and optimizers for large-scale training |

---

# 🎯 Repository Philosophy

Most resources teach:

> **"Here is a model."**

This repository teaches:

> **"Here is the engineering problem this model solves."**

Every chapter answers:

- Why was this technique invented?
- What problem does it solve?
- When should I use it?
- When should I avoid it?
- What replaced it?
- What are the trade-offs?
- What do production teams actually do?

The goal is to help you think like an experienced machine learning engineer—not simply memorize model names.

---

# 📂 Repository Structure

```text
docs/
│
├── part-1-training-strategy/
│   ├── README.md
│   ├── 01-transfer-learning.md
│   ├── 02-foundation-models.md
│   ├── 03-full-fine-tuning.md
│   ├── 04-lora.md
│   ├── 05-qlora.md
│   ├── 06-continued-pretraining.md
│   ├── 07-knowledge-distillation.md
│   └── 08-decision-framework.md
│
├── part-2-model-architectures/
│   └── README.md
│
├── part-3-generative-models/
│   └── README.md
│
├── part-4-training-optimization/
│   └── README.md
│
└── part-5-history-of-deep-learning/
    └── README.md
```

---

# 🤝 Contributing

Contributions are welcome.

Examples include:

- Improving explanations
- Adding production case studies
- Benchmark results
- Paper summaries
- Better visualizations
- Correcting technical inaccuracies

Please open an issue or submit a pull request.

---

# 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

# ⭐ About

**Deep Learning Training Playbook** transforms research papers into practical engineering playbooks.

Instead of reading dozens of papers before starting a project, use this repository to understand:

- **What works**
- **When it works**
- **Why it works**
- **What to choose for your own problem**

If you find this project useful, consider giving it a ⭐ to support its continued development.
