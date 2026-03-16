# GNR638 Assignments

This repository contains the implementations for assignments of the **GNR638 course**.

Each assignment is organized in a separate folder for clarity and ease of evaluation.

---

## Assignments

- [Assignment 1](Assignment1/) – Machine learning training and evaluation framework  
- [Assignment 2](Assignment2/) – To be added

---

## Repository Structure

```
GNR638_Assignments/
│
├── Assignment1/
│   ├── data/              # Data used for training and evaluation
│   ├── dataset/           # Dataset utilities and loaders
│   ├── framework/         # Core ML framework implementation
│   │
│   ├── train.py           # Script to train the model
│   ├── test.py            # Script to test the trained model
│   ├── eval.py            # Evaluation script
│   │
│   ├── config.json        # Configuration file containing hyperparameters
│   ├── setup.py           # Setup file for installing required modules
│   └── README.md          # Detailed instructions for Assignment 1
│
├── Assignment2/           # Implementation for Assignment 2
│
└── README.md              # Main repository documentation
```
```
GNR638_Assignments/
│
└── Assignment2/
    │
    ├── experiments/
    │   ├── corruption_test.py
    │   ├── few_shot.py
    │   ├── fine_tune.py
    │   ├── layer_probe.py
    │   ├── linear_probe.py
    │   └── layer_probe_log.txt
    │
    ├── models/
    │   └── model_loader.py
    │
    ├── training/
    │   ├── evaluate.py
    │   ├── train.py
    │   ├── train_finetune.py
    │   └── train_linear_probe.py
    │
    ├── utils/
    │   ├── corruptions.py
    │   ├── dataset.py
    │   ├── feature_extractor.py
    │   ├── metrics.py
    │   └── plots.py
    │
    ├── main.py
    ├── split_dataset.py
    ├── split_dataset_finetune.py
    ├── requirements.txt
    ├── .gitignore
    └── README.md
```
---

## Authors

Ananya Latchupatula  
Gehna Chelvi
