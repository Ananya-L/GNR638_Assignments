# GNR638 Assignments

This repository contains the implementations for assignments of the **GNR638 course**.

Each assignment is organized in a separate folder for clarity and ease of evaluation.

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
│
├── Assignment2/           # Implementation for Assignment 2
│
└── README.md              # Repository documentation
```

---

# Assignment 1

## Overview

This directory contains the implementation for **Assignment 1 of GNR638**.
The project provides a framework for training, testing, and evaluating machine learning models using the provided dataset and configuration files.

The repository includes scripts for:

* Training a model
* Testing the trained model
* Evaluating model performance
* Managing configurations through a JSON configuration file

---

## Directory Structure

```
Assignment1/
│
├── data/              # Data files used for training and evaluation
├── dataset/           # Dataset utilities and data loaders
├── framework/         # Core ML framework implementation
│
├── train.py           # Script to train the model
├── test.py            # Script to test the trained model
├── eval.py            # Script to evaluate model performance
│
├── config.json        # Configuration file containing hyperparameters
├── setup.py           # Setup script for installing modules
│
└── README.md          # Documentation
```

---

## Requirements

The project requires:

* Python **3.8 or higher**

Install required dependencies:

```
pip install numpy pandas torch scikit-learn tqdm
```

---

## Installation

Clone the repository:

```
git clone https://github.com/Ananya-L/GNR638_Assignments.git
```

Navigate to the Assignment 1 directory:

```
cd GNR638_Assignments/Assignment1
```

Install the project (optional):

```
pip install -e .
```

---

## Configuration

All model and training parameters are stored in:

```
config.json
```

This file contains parameters such as:

* learning rate
* batch size
* number of epochs
* dataset paths

Modify these parameters if necessary before running the scripts.

---

## Training the Model

To train the model run:

```
python train.py --config config.json
```

The trained model and logs will be saved in the output directory specified in the configuration file.

---

## Testing the Model

To run the trained model on the test dataset:

```
python test.py --config config.json
```

---

## Evaluation

To evaluate the model performance:

```
python eval.py --config config.json
```

Evaluation metrics will be printed to the console.

---

## Notes

* Ensure that the dataset is placed in the correct directory before running the scripts.
* Modify `config.json` if dataset paths or hyperparameters change.
* The scripts assume the directory structure shown above.

---

## Authors

**Ananya Latchupatula**

# Assignment 2

The implementation for Assignment 2 will be added in the `Assignment2` directory.

Instructions for compiling and running Assignment 2 will be provided inside that folder.

---

## Author

**Ananya Latchupatula, Gehna Chelvi**
