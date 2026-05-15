
## Introduction
* Source code for the paper "Diff-SelfA-PepGSI: A Diffusion-Based Framework for De novo Generation, Screening, and Morphological Identification of Self-Assembled Peptide Sequences".

* This study proposes a multi-stage generation and evaluation framework named Diff-SelfA-PepGSI, which decomposes the rational design of self-assembling peptides into three distinct, coupled functional modules: (1) a diffusion-based generation layer, which generates candidate sequences under the constraints of the target morphology; (2) a deep semantic screening layer, which utilises pre-trained protein language models and discriminative networks to achieve efficient sequence optimisation; and (3) a morphology classification layer, which employs ESM2, deep dual-branch feature extraction and discriminative networks to identify the morphological class to which the self-assembling sequences belong. 

<img width="3333" height="2500" alt="graph-abstract_300DPI" src="https://github.com/user-attachments/assets/9c2c8c8d-a72c-44f3-9c03-b6223e3a37b1" />
## Dataset
The data used in this paper (esol/lipo/freesolv) are publicly available on /data/.

## Environment
* base dependencies:
 ```bash
 pip install -r requirements.txt
 ```

## Usage

#### Quick Run
For ESOL:
```bash
python train.py configs/esol.json
```
For FreeSolv:
```bash
python train.py configs/freesolv.json
```
For Lipo:
```bash
python train.py configs/lipo.json
```
For AqSolDB:
```bash
python train.py configs/AqSolDB.json
```
For CASR-1:
```bash
python train.py configs/casr-1.json
```
