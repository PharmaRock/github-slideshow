---
layout: slide
title: "Preclinical → Clinical PK/PD Translation Workflow"
---

### Goal
Ingest **preclinical PK** for a molecule, enrich with **target antigen properties**, and predict **PD endpoints** against observed **clinical PK/PD**.

---

### Data to ingest
- Molecule attributes: modality, valency, Fc engineering, binding affinity.
- Preclinical PK: species, dose route, concentration-time profiles, NCA/compartmental parameters.
- Target antigen: baseline expression, turnover/internalization rate, soluble shed antigen, tissue distribution.
- Clinical readouts: PK profiles, biomarker/PD endpoints, receptor occupancy.

---

### Modeling approach
1. Build a **mechanistic PK/PD model** (TMDD or minimal PBPK + effect model).
2. Calibrate on preclinical data and antigen biology priors.
3. Apply allometric or physiology-based scaling into human.
4. Simulate clinical scenarios and compare predicted vs observed PK/PD metrics.

---

### Comparison outputs
- PK: Cmax, AUC, CL, Vd, t1/2.
- PD: Emax, EC50/IC50, time above threshold, target occupancy.
- Goodness-of-fit: prediction error, visual predictive checks, posterior predictive intervals.
