# DL-Playbook by wildoctopus 🐙
### A Researcher's Guide: Why to Use What, In Which Case

Stop guessing. This repo is a collection of decision frameworks for Deep Learning with evidence, code, and failure cases. 

Made for researchers who need to pick the right architecture, training strategy, and optimization trick for their specific problem.

---

## Core Philosophy
There is no "best model". There is only "best model *for your data, compute, and goal*".  
Every section here follows: `Problem -> Options -> Decision Rule -> Evidence -> Code`

---

## 1. Training Strategy: How Much To Fine-Tune?

Depends on 2 things: **Dataset Size** and **Domain Similarity**

![Transfer Learning Strategy Quadrant](assets/transfer-learning-quadrant.png)

| Case | Dataset | Use This | Why | Evidence |
| --- | --- | --- | --- | --- |
| **Large + Different** | 50k+ images, New domain | **Train Whole Model** | Pretrained features don't transfer. Need to relearn. | Kornblith 2019 |
| **Large + Similar** | 50k+ images, Same domain | **Freeze 65-70% Layers** | Keep low-level features, retrain high-level | Yosinski 2014 |
| **Small + Different** | <2k images, New domain | **Freeze 80-85% Layers** | Prevent overfitting. Only adapt head | Standard CV practice |
| **Small + Similar** | <2k images, Same domain | **Freeze Backbone, Train Head** | Pretrained = feature extractor | CLIP Linear Probe |

**Constants for all**: ImageNet `mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`. Use AdamW + CosineLR.

---

## 2. Architecture Choice: CNN vs Transformer vs MLP-Mixer

`Rule: It depends on data size and inductive bias needed`

| If you have... | Use This | Why | When NOT to use |
| --- | --- | --- | --- |
| **<10k images** | **CNN: ResNet, ConvNeXt** | Strong inductive bias: locality, translation equivariance | Don't use ViT. Will overfit |
| **10k - 1M images** | **Hybrid: Swin, ConvNeXt-V2** | Best of both. Local + Global attention | Avoid pure ViT without pretraining |
| **>1M images** | **Transformer: ViT, DeiT** | Scales best with data. No inductive bias needed | Waste of compute on small data |
| **Tabular + Image** | **MLP-Mixer / MLP** | Fast, simple. Good baseline | Poor for spatial tasks |

**Evidence**: Dosovitskiy 2020 "An Image is Worth 16x16 Words" - ViT needs 300M images to beat ResNet.

---

## 3. Optimization: Which Optimizer + Scheduler?

| Problem | Use This | Why |
| --- | --- | --- |
| **Training from scratch** | `SGD + Momentum 0.9 + StepLR` | Better generalization. Slower but stable |
| **Fine-tuning** | `AdamW + CosineAnnealingLR` | Adapts fast. Weight decay prevents overfit |
| **Diffusion / GenAI** | `AdamW 8-bit + LR 1e-4` | Memory efficient. Stable for UNet |
| **Training is unstable** | `Gradient Clipping + Warmup 500 steps` | Prevents loss spikes |

---

## 4. Data Problem: What Augmentation When?

| Dataset Size | Use This | Why |
| --- | --- | --- |
| **Large >50k** | `RandAugment, CutMix, MixUp` | Prevents overfitting, regularizes |
| **Small <5k** | `Basic: Flip, Crop, ColorJitter` | Aggressive aug will destroy signal |
| **Medical / Satellite** | `Domain-specific: Elastic Deform` | Generic aug breaks semantics |

---

## 5. GenAI Specific: Which Diffusion Model for What?

Since most of my research is here:

| Goal | Use This | Why |
| --- | --- | --- |
| **Virtual Try-On** | `IDM-VTON, OOTDiffusion` | Preserves garment details + person identity |
| **Image Editing** | `SDXL-Inpainting, BrushNet` | Best control + quality |
| **Fast Inference** | `LCM, SD-Turbo` | 4 steps instead of 50 |
| **Training on 1 GPU** | `LoRA, Dreambooth` | Only train 1% of parameters |

Full breakdowns in: `/genai/vton-papers/`

---

## 6. Common Mistakes Researchers Make

| Symptom | Root Cause | Fix from this repo |
| --- | --- | --- |
| Val loss goes up | Using ViT on 1k images | See Section 2: Use ResNet |
| Loss NaN in epoch 1 | LR too high for fine-tuning | See Section 3: Use 1e-5 |
| Model doesn't generalize | Wrong normalization | See Section 1: Use ImageNet stats |

---

## 7. How to Use This Repo
1.  Identify your problem: Data size, Domain, Goal
2.  Go to the relevant section above
3.  Copy the code + decision rule
4.  Cite the evidence paper

PRs welcome if you have a new "case -> solution" with a paper.

---

## About
Curated by **wildoctopus**. I break down SOTA research into decision trees.  
Other repos: [awesome-diffusion-vton](link) | [huggingface-cloth-segmentation](https://github.com/wildoctopus/huggingface-cloth-segmentation)

**License**: MIT
