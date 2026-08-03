# Chapter 10 — Choosing the Right Training Strategy
## The Engineering Decision Framework

> **Part I — Training Strategy**

---

**← Previous:** [Chapter 9 — Knowledge Distillation](09-knowledge-distillation.md)  
**📖 Part Home:** [Training Strategy](README.md)  
**🏠 Home:** [Deep Learning Training Playbook](../../README.md)

---

# Chapter Overview

| | |
|---|---|
| **Estimated Reading Time** | 45–60 minutes |
| **Difficulty** | ⭐⭐⭐⭐⭐ |
| **Prerequisites** | Chapters 1–9 |
| **Goal** | Learn how experienced ML engineers choose the right training strategy |

---

# The Most Important Question in This Repository

Throughout this guide we've studied many strategies.

- Transfer Learning
- Full Fine-Tuning
- LoRA
- QLoRA
- Continued Pretraining
- Knowledge Distillation

But knowing *what they are* isn't enough.

Real engineering begins with a different question.

> **Which one should I choose for my problem?**

This chapter answers that question.

---

# Think Like an ML Architect

Before writing a single line of training code, ask yourself these questions.

They should always be answered **in order**.

---

# Question 1 — Do You Really Need Training?

Many projects don't.

Ask yourself:

> **Does the model already perform well enough?**

If yes,

don't train.

If no,

continue.

---

## Example

You want an internal chatbot over company documentation.

The model already understands English.

It simply lacks access to your documents.

Training is unnecessary.

A retrieval system is likely the better solution.

---

# Question 2 — Is It a Knowledge Problem or a Behavior Problem?

This is one of the most common mistakes.

Ask:

> **Does the model lack knowledge, or does it behave incorrectly?**

Examples:

Knowledge problem:

- Medical terminology
- Banking regulations
- Scientific papers
- Proprietary APIs

Behavior problem:

- Output format
- Tone
- Instruction following
- JSON generation
- SQL generation

Knowledge problems usually require:

- Continued Pretraining
- RAG

Behavior problems usually require:

- Fine-Tuning
- LoRA
- QLoRA

---

# Question 3 — Does the Knowledge Change Frequently?

Suppose your documentation changes every week.

Training becomes expensive.

Instead ask:

> **Can I retrieve this information instead?**

| Knowledge Type | Better Choice |
|----------------|---------------|
| Company wiki | RAG |
| Daily financial reports | RAG |
| Product catalog | RAG |
| Medical language | Continued Pretraining |
| Programming language syntax | Pretraining |

A useful rule:

> **Dynamic knowledge belongs outside the model.**

---

# Question 4 — Does the Model Already Understand My Domain?

Examples:

General English?

Yes.

Medical pathology?

Maybe.

Satellite imagery?

Probably not.

Legacy COBOL?

Probably not.

If the answer is "no",

consider:

**Continued Pretraining**

before Fine-Tuning.

---

# Question 5 — How Much Labeled Data Do You Have?

| Dataset Size | Typical Strategy |
|--------------|------------------|
| <100 examples | Prompt Engineering / RAG |
| 100–5,000 | LoRA / QLoRA |
| 5k–100k | LoRA or Full FT |
| 100k+ | Benchmark Full FT |
| Millions | Consider training from scratch if no suitable foundation model exists |

Remember:

More data increases the value of updating more parameters.

---

# Question 6 — What Is Your GPU Budget?

| Hardware | Typical Recommendation |
|-----------|------------------------|
| Laptop GPU | Prompting, RAG |
| RTX 3090 / 4090 | QLoRA |
| Single A100 | LoRA / QLoRA |
| Multi-GPU Cluster | Full FT or Continued Pretraining |
| Large GPU Fleet | Any strategy based on benchmarking |

Never choose Full Fine-Tuning simply because it's possible.

Choose it because it provides measurable value.

---

# Question 7 — What Matters Most?

Every project optimizes something.

| Priority | Best Strategy |
|----------|---------------|
| Lowest training cost | QLoRA |
| Lowest inference cost | Distillation |
| Highest accuracy | Full FT |
| Fast experimentation | LoRA |
| Constantly changing knowledge | RAG |
| New domain | Continued Pretraining |

Every engineering decision is ultimately an optimization problem.

---

# The Master Decision Tree

```text
Need AI Model

        │

        ▼

Does Base Model Already Work?

        │

YES             NO

 │               │

Deploy      Knowledge
            Problem?

             │

      YES           NO

       │             │

Knowledge       Behavior

       │             │

Changes?      Domain
              Understood?

  │                │

YES              NO

 │                 │

RAG          Continued
             Pretraining

                   │

                   ▼

            Fine-Tuning

                   │

GPU Budget?

                   │

Low           High

 │              │

QLoRA      Full FT

      │

Need Faster Inference?

      │

YES

 │

Distillation

NO

Deploy
```

---

# Five Real Engineering Scenarios

## Scenario 1 — Enterprise Chatbot

You have:

- 20,000 internal PDFs
- Documents updated weekly

Choose:

✅ RAG

Not Fine-Tuning.

---

## Scenario 2 — Medical Diagnosis

You have:

- Millions of medical papers
- Large hospital datasets

Choose:

✅ Continued Pretraining

↓

Fine-Tuning

---

## Scenario 3 — Customer Support Assistant

You have:

- 15,000 labeled conversations
- One RTX 4090

Choose:

✅ QLoRA

---

## Scenario 4 — Mobile Image Classifier

Need:

- Small model
- Fast inference

Choose:

Teacher

↓

Knowledge Distillation

↓

Quantization

---

## Scenario 5 — Frontier AI Lab

Resources:

- Millions of GPUs
- Internet-scale dataset

Choose:

Pretraining

↓

Instruction Fine-Tuning

↓

RLHF / Preference Optimization

↓

Distillation

---

# Common Anti-Patterns

❌ Fine-Tuning because "everyone does it."

❌ Training when RAG is enough.

❌ Full FT on 500 examples.

❌ Continued Pretraining on tiny datasets.

❌ Distilling a poor teacher.

❌ Assuming larger models always solve the problem.

Engineering is about solving the correct problem,

not applying fashionable techniques.

---

# The Golden Rules

1. Start with the simplest solution.
2. Benchmark before scaling.
3. Data quality beats model size.
4. Dynamic knowledge belongs in retrieval systems.
5. Train only when training adds measurable value.
6. Deployment constraints matter as much as benchmark scores.
7. Every additional GPU hour should have a business justification.

---

# One Mental Model to Remember

Every training strategy answers a different question.

| Question | Strategy |
|----------|----------|
| Should I reuse an existing model? | Transfer Learning |
| Should I update every parameter? | Full Fine-Tuning |
| Can I update only a few parameters? | LoRA |
| Can I fit it on my GPU? | QLoRA |
| Does the model understand my domain? | Continued Pretraining |
| Is deployment too expensive? | Knowledge Distillation |
| Does the knowledge change frequently? | RAG |

When you know the question,

the answer usually becomes obvious.

---

# Engineering Summary

Machine learning is not about using the newest technique.

It is about selecting the **least expensive approach that reliably solves the problem.**

The best engineers are not the ones who know the most algorithms.

They are the ones who know **when not to use them.**

That is the purpose of this playbook.

---

# References

This chapter builds upon the techniques discussed throughout Part I.

Readers are encouraged to revisit the individual chapters for mathematical details, implementation guidance, and original research papers.

---

# Congratulations

You have completed **Part I — Training Strategy**.

You now understand not only *how* modern training techniques work,

but more importantly,

**when they should be used.**

The next part of the playbook moves from **training decisions** to **model architecture decisions**, answering questions such as:

- CNN or Vision Transformer?
- ConvNeXt or SigLIP?
- Transformer or Mamba?
- Encoder or Decoder?
- Diffusion or Flow Matching?

These decisions determine *what kind of model* you should build before training even begins.
