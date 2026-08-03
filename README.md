[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# Deep Learning Training Playbook

Practical strategies, experiments, and lessons learned for training models from **2014 to 2026**.

This guide helps you answer **"what to use, when, and how"** so you don't waste GPU and time.

---

## 🤔 Table of Contents - Ask Yourself:

1.  **[How do I choose a training strategy?](docs/transfer-learning.md)**  
    `From scratch vs Fine-tune vs LoRA` → See the 4-quadrant rule
2.  **[Which architecture should I pick for my problem?](#q1-which-architecture-should-i-use)**  
    `Medical vs Web-scale vs Long context vs Multimodal`
3.  **[Which generative model should I use?](#q2-which-generative-model-should-i-use)**  
    `T2I vs LoRA vs Video vs Virtual Try-On`
4.  **[How do I train faster and use less GPU?](#q3-how-do-i-train-faster-and-use-less-gpu)**  
    `QLoRA vs Muon vs Lion vs AdamW`
5.  **[Why did we end up with these models?](#part-5-how-we-got-here---2012-to-2026)**  
    `The history from AlexNet to Flux/Mamba2`

---

## Part 1: The Timeless Rule - Transfer Learning

> **Before 2014 everyone trained from scratch.**  
> *Expensive, slow, and $10k per model.*

Researchers found there are only **4 cases** that still rule in 2026:

[Transfer Learning Strategy Quadrant](https://raw.githubusercontent.com/wildoctopus/effective-deep-learning-training-strategies/main/assets/imgclass.svg)

**The core question this answers:**  
`Do I have enough data?` + `Is my data similar to ImageNet?` = **Pick 1 of 4 strategies**

**How to read this:**
- **X-axis: Domain Similarity** → `Cats/Dogs = Similar to ImageNet` | `Xray/Satellite = Different`
- **Y-axis: Dataset Size** → `1k images = Small` | `1M images = Large`

> **2026 Update**: The logic is still the same.  
> "Train Remaining" now just means **"use LoRA/QLoRA"** to save 10x GPU.

[**→ Read the full Transfer Learning Deep Dive**](docs/transfer-learning.md)  
*Covers: when to freeze, when to full-finetune, domain gap math, and 4 real case studies*

---

## Part 2: Quick Questions & Answers

### Q1: **Which architecture should I use?**

| **Problem You're Solving** | **Best Architecture 2026** | **Why This Wins** | **Watch Out** |
| :--- | :--- | :--- | :--- |
| 🩺 **Medical, Satellite, Industrial** <br> *Small data, need accuracy, can't overfit* | **`ConvNeXtV2`** | Has inductive bias built in. Learns local features fast. <br> **Gets to 90% with 5k images** | Hits ceiling at 10M+ images. Worse than ViT at scale |
| 🌐 **Web-scale Classification, CLIP** <br> *Millions of images, need SOTA* | **`SigLIP2`, `ViT-G/14`** | No bias = scales better. Sees global context. <br> **+3% over ConvNeXt at 50M images** | Needs 1M+ images or it overfits. 24GB+ VRAM |
| 📄 **Long Docs, Code Repos, Books** <br> *Context > 32k tokens* | **`Mamba2`, `RWKV-6`** | `O(n)` memory. Processes like "passing notes". <br> **1M context on 24GB GPU** | 3-5% worse reasoning than GPT-4o. New ecosystem |
| 🖼️ **VLM, OCR, Video Understanding** <br> *Images + Text together* | **`Qwen2-VL`, `Gemma3`** | Trained natively on both modalities. <br> **Best OCR + grounding in 2026** | 2x slower and 2x VRAM vs text-only LLM |

[**→ Read the full Architecture Deep Dive**](docs/architecture.md)

### Q2: **Which generative model should I use?**

| **Problem You're Solving** | **Best Model 2026** | **Why This Wins** | **Watch Out** |
| :--- | :--- | :--- | :--- |
| 🖌️ **Product Ads, Posters** <br> *Need perfect text + follow prompts* | **`Flux.1-dev`, `SD3.5`** | Rectified Flow. 20 steps. <br> **Only model that spells correctly** | 12GB+ VRAM. ~8s per image on 3090 |
| 👕 **E-commerce, Virtual Try-On** <br> *Put clothes on people, keep details* | **`CatVTON`, `Leffa`** | No pose estimation needed. <br> **Preserves fabric texture + hands** | Only works on humans. Not for products |
| 🎭 **Brand Consistency, Characters** <br> *Train on 50-200 images* | **`Flux LoRA`, `SD3 LoRA`** | Trains 0.1% of weights. <br> **30 min on 1 GPU, doesn't break base model** | Won't do full style transfer. Use DreamBooth for that |
| 🎬 **Short Videos, Ads** <br> *5-10s clips with motion* | **`Kling`, `Runway Gen3`** | Best temporal coherence in 2026. <br> **Less flickering, better physics** | $ per second. Still not perfect with hands |

[**→ Read the full Generative Deep Dive**](docs/generative.md)

### Q3: **How do I train faster and use less GPU?**

| **Problem You're Solving** | **Best Setup 2026** | **Why This Wins** | **Watch Out** |
| :--- | :--- | :--- | :--- |
| 🤏 **Fine-tune 7B LLM on 3090** <br> *Can't afford A100s* | **`QLoRA r=64` + `8-bit AdamW`** | 4-bit base + adapters. <br> **Fits in 20GB VRAM, 90% of full FT quality** | 2-3% accuracy drop. Slower training |
| 🚀 **Train 1B+ Model from Scratch** <br> *Need to save GPU $* | **`Muon` + `Sophia` + `WSD LR`** | 2nd order optimizer. <br> **~1.8x faster convergence vs AdamW** | Overhead on <1B models. New, less stable |
| 💾 **Fine-tune ViT on 1 GPU** <br> *Memory bottleneck* | **`Lion` optimizer** | Half the optimizer states. <br> **Same accuracy, 50% less VRAM** | Needs LR tuning. Not as robust as AdamW |
| ✅ **Just want it to work** <br> *90% of projects* | **`AdamW` + `Cosine Decay`** | Battle tested. <br> **Works on everything** | Not the fastest. Not the most memory efficient |

[**→ Read the full Optimization Deep Dive**](docs/optimization.md)

---

## Part 5: How We Got Here - 2012 to 2026

| **Era** | **Key Breakthrough** | **Why it Mattered** |
| :--- | :--- | :--- |
| **2012** | `AlexNet` | Proved **deep CNNs work** |
| **2014** | Transfer Learning | **Reuse ImageNet features**. The Quadrant above |
| **2017** | `Transformer` | *Every token can talk to every other token* |
| **2020** | `CLIP`, `GPT-3` | **Pretrain once**, adapt to anything |
| **2023** | `SDXL` | GANs died. **Image quality jumped** |
| **2024** | `LoRA`, `QLoRA` | Fine-tune **70B models on 1 GPU** |
| **2026** | `Mamba2`, `Flux`, `Muon` | **Faster, cheaper, and better** than 2023 |

---

## License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## About
Turning **research papers into decision trees** so you don't have to read 50 pages to pick a model.

PRs and benchmarks welcome. **This is a living guide.**
