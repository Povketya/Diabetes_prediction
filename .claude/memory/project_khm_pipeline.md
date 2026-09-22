---
name: project-khm-pipeline
description: Cambodia STEPS diabetes prediction pipeline — data flow, notebook structure, and verified feature mappings
metadata:
  type: project
---

The Cambodia STEPS diabetes prediction pipeline is split across four notebooks in `notebooks/`:

1. **khm_preprocessing.ipynb** — `steps_selected_before_ai_generation.csv` (873×36) → `cambodia_diabetes_final_clean.csv` (873×14)
2. **khm_splitting.ipynb** — stratified 70/15/15 real-only splits → `train.csv` (611), `validation.csv` (131), `test.csv` (131)
3. **khm_ctgan.ipynb** — CTGAN from `train.csv` → `cambodia_ai_final_clean.csv` (5000 synthetic) + `train_ready.csv` (5611 combined)
4. **khm_training.ipynb** — 7 classifiers + soft-voting ensemble, threshold tuning, final test evaluation

All files live in `newdata/`. CTGAN is not in requirements.txt — notebook installs it at runtime.

**Verified STEPS → feature mappings (confirmed 100% accurate):**
- `diabetes_binary` ← h7a (1→1, 2→0)
- `sex` ← sex (1→1, 2→0)
- `age` ← age (kept as-is)
- `education` ← c5 (1–7 scale, directly encoded)
- `bmi` ← m4 / (m3/100)² rounded to 1 dp; median-imputed
- `highbp` ← (h1==1) AND (h2a==1) — self-reported doctor diagnosis
- `highchol` ← b8.notna() — blood glucose proxy (no cholesterol in dataset)
- `smoker` ← t1==1
- `physactivity` ← NOT (p1==p4==p7==p10==p13==2) — any PA module active
- `fruits` ← d1>0
- `veggies` ← d3>0
- `sedentaryminutes` ← p16a×60 + p16b
- `waist_cm` ← m7
- `hvyalcoholconsump` ← (a3==1) AND (a5>=4) — ~95% accurate (binge ≥4 times/month)

**Why:** Produced to document and reproduce the Cambodia cohort preprocessing for the final year project report.
**How to apply:** When working on KHM preprocessing or feature questions, these are the exact formulas used.
