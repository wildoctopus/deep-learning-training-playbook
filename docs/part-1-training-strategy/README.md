# Part I — Training Strategy
## How Modern Deep Learning Models Are Trained, Adapted, and Deployed

> *"Training a model is easy. Choosing the right training strategy is the hard part."*

---

**🏠 Home:** [Deep Learning Training Playbook](../../README.md)

---

# Why This Section Exists

Every year, new training techniques emerge:

- Full Fine-Tuning
- LoRA
- QLoRA
- Continued Pretraining
- Knowledge Distillation
- Preference Optimization
- Reinforcement Learning

Most articles explain **how** these methods work.

Very few explain:

> **When should I use each one?**

This section is designed to answer exactly that question.

Instead of presenting isolated techniques, we'll build a mental model that helps you make engineering decisions based on:

- Dataset size
- Domain similarity
- Compute budget
- GPU memory
- Deployment constraints
- Business requirements

By the end of this section, you'll know not only *how* these techniques work, but also *why* they exist and *when* they should be used.

---

# Learning Journey

This section is written as a story.

Each chapter answers one engineering question, leading naturally to the next.

```text
Training From Scratch

        │

        ▼

Transfer Learning

        │

        ▼

Foundation Models

        │

        ▼

Need Adaptation?

        │

        ├──────────────┐
        ▼              ▼

Full FT          Parameter Efficient
                      │
                      ▼
                    LoRA
                      │
                      ▼
                    QLoRA

        │

Need New Domain Knowledge?

        │

        ▼

Continued Pretraining

        │

Need Cheap Deployment?

        │

        ▼

Knowledge Distillation

        │

        ▼

Engineering Decision Framework
```

---

# Reading Order

## Chapter 1 — Why Transfer Learning Changed Everything

Understand why the industry moved away from training models from scratch.

**You'll learn**

- The history of transfer learning
- Why ImageNet changed deep learning
- The four-quadrant strategy
- The importance of dataset size and domain similarity

➡ **Read:** [01-transfer-learning.md](01-transfer-learning.md)

---

## Chapter 2 — Foundation Models

Learn why today's AI starts with pretrained foundation models.

**You'll learn**

- Self-supervised learning
- Scaling laws
- Why GPT, CLIP and ViT changed AI
- Why pretraining is expensive but reusable

➡ **Read:** [02-foundation-models.md](02-foundation-models.md)

---

## Chapter 3 — Full Fine-Tuning

Understand what happens when every parameter is updated.

**You'll learn**

- End-to-end optimization
- Memory requirements
- Advantages and limitations
- When Full FT is worth the cost

➡ **Read:** [03-full-fine-tuning.md](03-full-fine-tuning.md)

---

## Chapter 4 — LoRA

Learn how to adapt billion-parameter models by training only a tiny fraction of the weights.

**You'll learn**

- Low-rank adaptation
- Matrix decomposition intuition
- Parameter-efficient fine-tuning
- Why LoRA became the industry standard

➡ **Read:** [04-lora.md](04-lora.md)

---

## Chapter 5 — QLoRA

Compress the frozen model while training lightweight adapters.

**You'll learn**

- 4-bit quantization
- NF4
- Double Quantization
- Paged Optimizers
- Consumer GPU fine-tuning

➡ **Read:** [05-qlora.md](05-qlora.md)

---

## Chapter 6 — Continued Pretraining

Teach a model your domain before teaching it your task.

**You'll learn**

- Domain-Adaptive Pretraining (DAPT)
- Task-Adaptive Pretraining (TAPT)
- Continued Pretraining vs RAG
- Building domain-specific foundation models

➡ **Read:** [06-continued-pretraining.md](06-continued-pretraining.md)

---

## Chapter 7 — Knowledge Distillation

Transfer knowledge from a large model into a smaller one.

**You'll learn**

- Teacher–Student training
- Soft labels
- Temperature scaling
- Distillation for LLMs
- Production deployment

➡ **Read:** [07-knowledge-distillation.md](07-knowledge-distillation.md)

---

## Chapter 8 — Engineering Decision Framework

Bring everything together.

**You'll learn**

- Which strategy to choose
- Common engineering mistakes
- Decision trees
- Production case studies
- Practical playbook for real projects

➡ **Read:** [08-decision-framework.md](08-decision-framework.md)

---

# What You Will Be Able to Do

After completing this section, you should be able to answer questions like:

- Should I train from scratch?
- Should I use Transfer Learning?
- Should I use Full Fine-Tuning or LoRA?
- When is QLoRA enough?
- When do I need Continued Pretraining?
- Should I use RAG instead of training?
- When should I distill my model?
- Which strategy minimizes cost while maximizing performance?

More importantly, you'll understand **why** each answer is correct.

---

# Who Should Read This?

This section is intended for:

- Machine Learning Engineers
- AI Engineers
- Deep Learning Researchers
- Software Engineers moving into AI
- MLOps Engineers
- Technical Leads
- Students preparing for AI interviews

If you've ever asked:

> *"I know these techniques exist, but I don't know when to use them."*

this section is for you.

---

# Key Papers Covered

The ideas in this section are grounded in influential research papers, including:

- AlexNet (2012)
- ImageNet Transfer Learning
- BERT (2018)
- GPT-3 (2020)
- CLIP (2021)
- LoRA (2021)
- Don't Stop Pretraining (2020)
- QLoRA (2023)
- Knowledge Distillation (2015)

Each chapter explains not just the paper's results, but the engineering decisions that followed.

---

# Where to Go Next

Once you've learned **how to train models**, the next question becomes:

> **Which architecture should I choose?**

Continue to:

➡ **Part II — Model Architectures**

There you'll learn:

- CNN vs Vision Transformer
- ConvNeXt vs ViT
- CLIP vs SigLIP
- Transformer vs Mamba
- Encoder vs Decoder
- Multimodal foundation models
- Architecture decision frameworks

---

> **"The best training strategy cannot save the wrong architecture.  
> The best architecture cannot compensate for the wrong training strategy.  
> Great AI systems require getting both right."**
