# Flexible Learning PLM Framework  
_Active / Reinforcement‑Learning on Top of ESM‑like Embeddings_

---

## 1 Vision & Scope
**Goal.**  Enable rapid, low‑compute optimization of protein sequences by coupling pretrained Protein‑Language‑Model (PLM) embeddings (e.g. Facebook Research ESM) with a lightweight learner that can ingest **0‒few rounds** of experimental variant data.

**Key capabilities**
* **Plug‑and‑play models.**  Accept any ESM checkpoint; default to `esm2_t33_650M`.
* **Two learning modes.**  
  * **Active‑learning head** – Bayesian linear/MLP or Ridge‑regression with uncertainty estimates.  
  * **Reinforcement learning (RL) head** – policy‑gradient over token edits with reward = predicted function.
* **Round‑based workflow.**  Users may (i) upload all assayed variants at once, or (ii) iterate: _propose → assay → update_.
* **Proposal controls.**  Batch size, temperature (Softmax over acquisition scores), mutation budget, constrained positions.
* **Compute-class friendly.**  Entire training loop < 2 h on 8‑core CPU (freeze PLM; fine‑tune ≤ 2 adapter layers or LoRA ranks = 4).

---

## 2 System Architecture
```text
┌───────────────────────┐
│   Data Manager        │ ←─ CSV / FASTA / JSON
└────────┬──────────────┘
         │(batched)
┌────────▼──────────────┐      ┌────────────────────────┐
│  ESM Encoder          │      │  Proposal Engine       │
│ (frozen weights)      │──▶──▶│  (Active / RL policy)  │─┐
└────────┬──────────────┘      └────────────────────────┘ │
         │embeddings                                 picks│
┌────────▼──────────────┐                                 │
│  Learner Head         │◀────────────────────────────────┘
│  (adapter / linear)   │   reward = assay data / proxy
└───────────────────────┘
```

* **Storage layer** – lightweight SQLite or DuckDB for assayed variants, metadata, acquisition scores.  
* **Interface** – Python API + CLI; optional Streamlit dashboard.  
* **Extensibility** – all modules registered via entry‑points (`plugins/*`).

---

## 3 Algorithms & Methods
| Component | Default | Alternatives |
|-----------|---------|--------------|
| **Embeddings** | ESM‑2 650 M, residue‑level pooled to sequence | ProtGPT, ProGen2 |
| **Learner** | Ridge regression (active) or A2C (RL) on top of 1024‑D reduced embeddings (PCA‑128) | GP‑Bayesian, Dueling DDQN |
| **Acquisition** | Upper Confidence Bound (μ + k·σ) | EI, Thompson samp., Diversity‑weighted |
| **Reward shaping (RL)** | `scaled_function_score – λ·(#mutations)` | In‑silico proxy (UniRep‑ΔΔG) |

---

## 4 Evaluation & Benchmarks
* **Held‑out assays** from Science _2024_ Dallago _et al._ (adr6006) & Nat Commun 2025 dataset 55987‑8.  
* **Metrics:** Spearman ρ, top‑k‑hit‑rate @{10,50}, & experimental budget efficiency (#assays to reach 90 % of best variant).  
* **Ablations** – head type, #trainable layers {0,1,2}, acquisition strategy.

---

## 5 Risks & Mitigations
| Risk | Mitigation |
|------|------------|
| Over‑fitting on ≤ 32 variants | strong L2, early stopping, bagging |
| Embedding drift across PLM versions | version‑pinned cache, unit tests |
| Compute blow‑up with RL |  token‑budget limit + entropy regularization |

---

## 6 Milestones & Deliverables
1. **v0.1** – CLI to embed sequences & train ridge head on static dataset.  
2. **v0.2** – Active‑learning loop with UCB acquisition.  
3. **v0.3** – RL head prototype (A2C) with toy reward.  
4. **v1.0** – Full plugin framework, docs, CI, PyPI release.

---

## 7 Tech Stack
* **Python ≥ 3.10**, PyTorch 2.x, HuggingFace Transformers, `bitsandbytes` (CPU fallback).  
* Lightweight: scikit‑learn, Ray Tune (optional), Pydantic, Typer, Streamlit.  
* CI/CD: GitHub Actions, Poetry, pre‑commit (ruff, black).

---
