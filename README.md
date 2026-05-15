
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
python gan_diff.py
```
```bash
python SelfA_Classification.py
```
```bash
python Identification-classification.py
```

### 5. Generate sequence

```bash
# Generate unconditionally
python gan_diff_generate.py \
  --MAX_SEQ_LEN = 6 \
  --num_outputs = 20000 \
  --output_file = "seq_len6.txt"\
```

### 6. Screening and Identification

```bash
# Screening
python SelfA_Classification_Prediction.py \
  --input_file = "seq_len6.txt"\
  --output_file = "seq_len6_.txt"\
  --pos_file = "pos_len6.txt"\
```
```bash
# Identification
python Identification-prediction.py \
  --input_file = "pos_len6.txt"\
  --output_file = "result_len6.txt"\
```
### 7. Retrieve the top 20 sequences by confidence score for each category

```bash
# Screening
python confidence-max.py \
  --input_file = "result_len6.txt"\  
  --output_file = "top20_per_class_len6.txt"  
```




---
