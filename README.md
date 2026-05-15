
## Introduction
* Source code for the paper "Diff-SelfA-PepGSI: A Diffusion-Based Framework for De novo Generation, Screening, and Morphological Identification of Self-Assembled Peptide Sequences".

* This study proposes a multi-stage generation and evaluation framework named Diff-SelfA-PepGSI, which decomposes the rational design of self-assembling peptides into three distinct, coupled functional modules: (1) a diffusion-based generation layer, which generates candidate sequences under the constraints of the target morphology; (2) a deep semantic screening layer, which utilises pre-trained protein language models and discriminative networks to achieve efficient sequence optimisation; and (3) a morphology classification layer, which employs ESM2, deep dual-branch feature extraction and discriminative networks to identify the morphological class to which the self-assembling sequences belong. 

<img width="3333" height="2500" alt="graph-abstract_300DPI" src="https://github.com/user-attachments/assets/9c2c8c8d-a72c-44f3-9c03-b6223e3a37b1" />


## 🚀 Quick strat

### 1. environment install

* base dependencies:
 ```bash
 pip install -r requirements.txt
 ```

### 2. Data prepration

The data used in this paper (esol/lipo/freesolv) are publicly available on Diff-SelfA-PepGSI/data/.

Dataset1 in paper: data_all.csv; Dataset2 in paper: data_all_seq_and_label.csv

Data Format for Dataset1：

| List | Type | Note | Example |
|------|------|------|------|
| Seq | string | Amino acid sequence | YYKLVFFC |
| Label | int | Self-assembly propertie | 0/1 |

Data Format for Dataset2：

| List | Type | Note | Example |
|------|------|------|------|
| Seq | string | Amino acid sequence | POGPOGPOGPOGPAGPOGPOGPOGPOGPOG |
| Label | string | Morphological information | fiber |


### 3. Split the dataset

```bash
python 5-Fold-Screening.py
```
```bash
python 5-Fold-Identification.py
```

### 4. Train model

```bash
python train.py --config configs/default.yaml
```

### 5. Generate sequence

```bash
# Generate unconditionally
python generate.py \
  --checkpoint checkpoints/best_model.pt \
  --num_samples 100 \
  --seq_len 30 \
  --guidance_scale 3.0 \
  --output_dir results/unconditional

# Generation of fixed secondary structures
python generate.py \
  --checkpoint checkpoints/best_model.pt \
  --fix_structure \
  --target_ss "H" \
  --num_samples 50 \
  --output_dir results/fixed_structure

# Generation of objective constraints
python generate.py \
  --checkpoint checkpoints/best_model.pt \
  --target_charge 5.0 \
  --target_hydro -0.3 \
  --num_samples 50 \
  --seq_len 25 \
  --output_dir results/target_properties
```

---
