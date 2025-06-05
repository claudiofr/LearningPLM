# Task List – Flexible Learning PLM Framework
_Grouped by work‑stream; tick ☑︎ when done._

---

## 0 ‒ Project Bootstrap
- [x] Create Git repo; enable GitHub Actions template.  
- [x] Define editable Poetry environment (`pyproject.toml`).  
- [ ] Pre‑commit hooks: ruff, black, isort, mdformat.

---

## 1 ‒ Core Infrastructure
- [x] **Data schema** → `datamodels.py` (Pydantic) for Variant, AssayResult, RoundMetadata.  
- [x] **Data manager** with SQLite / DuckDB backend; CRUD, versioning.  
- [x] **Embedding cache** with mmap HDF5; script `embed.py` (batch, multiprocess).  
- [x] **Config system** via `yaml` + env overrides (OmegaConf).

---

## 2 ‒ Learner Heads
- [x] Ridge / MLP head using `sklearn` / `torch.nn`.  
- [ ] Adapter‑LoRA utility to unfreeze ≤ 2 layers (`lora_r=4`).  
- [x] RL policy (A2C): action = {pos, aa}, mask illegal edits.

---

## 3 ‒ Active‑Learning Loop
- [x] Acquisition strategies module: UCB, EI, Thompson, diversity‑penalized.  
- [x] `controller.py` orchestrating **propose → assay → fit** cycle.  
- [x] Temperature & batch‑size scheduling CLI flags.

---

## 4 ‒ User Interfaces
- [x] **CLI** (Typer): `plm propose`, `plm learn`, `plm benchmark`.  
- [x] **Python API**: `from plm import Framework; Framework().fit_round(...)`.  
- [ ] **Dashboard (optional)**: Streamlit view of embedding UMAP & acquisition scores.

---

## 5 ‒ Evaluation & Benchmarks
- [ ] Port Science 2024 & Nat Commun 2025 datasets (convert to internal schema).  
- [ ] `scripts/benchmark.py` to generate metrics table & figures.  
- [ ] Unit tests (> 80 % coverage) for head training & acquisition math.

---

## 6 ‒ Performance & Optimization
- [ ] FP16 / Int8 quantization toggle (`bitsandbytes`).  
- [ ] CPU‑only training path; profiling with `torch.profiler`.  
- [ ] Automated runtime budget check (< 2 h on 8‑core Intel i7).

---

## 7 ‒ Documentation & Examples
- [ ] MkDocs site with tutorial notebooks ("0‑shot", "1‑round", RL demo).  
- [ ] API reference auto‑generated via `mkdocstrings`.  
- [ ] Contribution guide & code‑of‑conduct.

---

## 8 ‒ Release Engineering
- [ ] Semantic versioning; generate changelog with `towncrier`.  
- [ ] Build & publish wheels to TestPyPI → PyPI.  
- [ ] Dockerfile (CPU slim) for reproducible runs.

---

## 9 ‒ Stretch Goals
- [ ] Multi‑objective optimization (e.g., stability × activity via Pareto front).  
- [ ] GPT‑4o‑driven candidate filtering (natural‑language constraints).  
- [ ] Fine‑tune on metagenomic unlabeled sequences (self‑distillation).

---
