import torch
import pandas as pd

from models.model_loader import load_model
from utils.dataset import get_dataloaders
from utils.corruptions import gaussian_noise, motion_blur, brightness_shift
from training.evaluate import evaluate


device = "cuda" if torch.cuda.is_available() else "cpu"

_, val_loader, num_classes = get_dataloaders("dataset")

models = [
    "resnet50",
    "densenet121",
    "efficientnet_b0"
]

corruptions = {
    "clean": lambda x: x,
    "gaussian_noise": gaussian_noise,
    "brightness_shift": brightness_shift,
    "motion_blur": motion_blur
}

results = []

for model_name in models:

    model = load_model(model_name, num_classes)
    model.load_state_dict(
        torch.load(f"/content/drive/MyDrive/GNR638_results/{model_name}_fewshot_model.pth")
    )

    model = model.to(device)
    model.eval()

    for cname, corrupt in corruptions.items():

        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in val_loader:

                images = images.to(device)

                images = torch.stack([corrupt(img) for img in images])

                outputs = model(images)

                preds = outputs.argmax(1)

                correct += (preds.cpu() == labels).sum().item()

                total += labels.size(0)

        acc = correct / total

        results.append({
            "model": model_name,
            "corruption": cname,
            "accuracy": acc
        })

df = pd.DataFrame(results)

df.to_csv("corruption_results.csv", index=False)

print(df)