import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt

from models.model_loader import load_model
from utils.dataset import get_dataloaders, get_subset_loader
from training.train import train_one_epoch
from training.evaluate import evaluate


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"

train_loader, val_loader, num_classes = get_dataloaders("dataset")

dataset = train_loader.dataset

fractions = [1.0, 0.2, 0.05]

models_to_run = [
    "resnet50",
    "densenet121",
    "efficientnet_b0"
]

all_results = {}

for model_name in models_to_run:

    print(f"\n================ {model_name} =================\n")

    results = {}

    for frac in fractions:

        print(f"\n===== Training with {int(frac*100)}% data =====")

        subset = get_subset_loader(dataset, frac)

        loader = torch.utils.data.DataLoader(
            subset,
            batch_size=32,
            shuffle=True
        )

        model = load_model(model_name, num_classes)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        epochs = 20

        for epoch in range(epochs):

            train_loss, train_acc = train_one_epoch(
                model,
                loader,
                optimizer,
                criterion,
                device
            )

        val_acc = evaluate(model, val_loader, device)

        gap = train_acc - val_acc

        print(f"Train Accuracy: {train_acc:.3f}")
        print(f"Validation Accuracy: {val_acc:.3f}")
        print(f"Training–Validation Gap: {gap:.3f}")

        results[frac] = (train_acc, val_acc)

    all_results[model_name] = results

    print("\nFinal Results:")

    for k, v in results.items():
        train_acc, val_acc = v
        print(f"{int(k*100)}% data → Train: {train_acc:.3f}, Val: {val_acc:.3f}")

    acc100 = results[1.0][1]
    acc5 = results[0.05][1]

    delta = (acc100 - acc5) / acc100

    print("Relative Performance Drop:", delta)

    fractions_percent = [100, 20, 5]
    accuracies = [
        results[1.0][1],
        results[0.2][1],
        results[0.05][1]
    ]

    plt.figure()

    plt.plot(fractions_percent, accuracies, marker='o')

    plt.xlabel("Training Data (%)")
    plt.ylabel("Validation Accuracy")
    plt.title(f"Few-Shot Learning Performance - {model_name}")

    plt.savefig(f"few_shot_{model_name}.png")

    plt.show()