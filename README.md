# BDFM

This repository provides a masked image modeling (MIM)-based foundation model of brain disease (BDFM), which was trained on 15,300 brain MRI scans. The training and fine-tuning processes are illustrated below.

## Requirements

```bash
pip install -r requirements.txt
```

## Datasets (BD-15k database)
The details of the pre-training and fine-tuning datasets are shown in the following figure. These datasets are publicly accessible through their respective official websites.

[![Dataset Overview](Datasets.png)](Datasets.png)

```bash
datasets/
├── BraTS23_GLI
│   ├── TrainingData
│   └── ValidationData
├── BraTS23_MEN
│   ├── TrainingData
│   └── ValidationData
├── BraTS23_MET
│   ├── TrainingData
│   └── ValidationData
├── BraTS23_PED
│   ├── TrainingData
│   └── ValidationData
├── BraTS23_SSA
│   ├── TrainingData
│   └── ValidationData
├── AtlasR2
│   ├── TrainingData
│   └── ValidationData
├── BrainPTM2021
│   ├── TrainingData
│   └── ValidationData
├── ISLES2022
│   ├── TrainingData
│   └── ValidationData
├── OASIS
│   ├── TrainingData
│   └── ValidationData
├── MRBrains13
│   ├── TrainingData
│   └── ValidationData
└── UPENN-GBM
    ├── TrainingData
    └── ValidationData
```

After downloading the data, generate JSON files:
```bash
get_json.py
```
## Pretrain

