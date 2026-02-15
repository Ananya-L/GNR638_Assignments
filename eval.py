import argparse
import pickle
import json
import cv2
import os

from framework.layers import Conv2D, ReLU, MaxPool2D, Flatten, Linear
from framework.tensor import Tensor


# ------------------------------------------------------------
# LOAD SAVED MODEL
# ------------------------------------------------------------
def load_model(params, filepath):
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)

    loaded_params = model_data['parameters']

    for p, lp in zip(params, loaded_params):
        p.data = lp

    print(f"Model loaded from {filepath}")


# ------------------------------------------------------------
# DATASET LOADER (IDENTICAL TO TRAINING PREPROCESSING)
# ------------------------------------------------------------
class ImageDataset:
    def __init__(self, root_dir, image_size=(32, 32)):
        self.images = []
        self.labels = []
        self.class_map = {}

        print("Loading dataset from:", root_dir)

        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_path):
                continue

            self.class_map[class_name] = idx

            for fname in os.listdir(class_path):
                if not fname.lower().endswith(".png"):
                    continue

                img_path = os.path.join(class_path, fname)

                img = cv2.imread(img_path)

                # ---- SAME AS TRAINING ----
                img = cv2.resize(img, image_size)
                img = img.astype("float32") / 255.0
                img = img.transpose(2, 0, 1)  # HWC → CHW
                # --------------------------

                self.images.append(img.tolist())
                self.labels.append(idx)

        print(f"Found {len(self.images)} images in {len(self.class_map)} classes")
        print("Class mapping:", self.class_map)

    def __len__(self):
        return len(self.images)

    # correct batching (keeps last batch)
    def get_batches(self, batch_size):
        for start in range(0, len(self.images), batch_size):
            end = start + batch_size

            batch_x = self.images[start:end]
            batch_y = self.labels[start:end]

            yield Tensor(batch_x, requires_grad=False), batch_y


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--config', type=str, default='config.json')
    parser.add_argument('--weights_path', type=str, default='model.pkl')
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()

    # ---------------- CONFIG ----------------
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except:
        config = {'conv_channels': 16, 'conv_kernel': 3}

    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)

    dataset = ImageDataset(args.dataset_path)
    num_classes = len(dataset.class_map)

    print(f"Total samples: {len(dataset)}")
    print(f"Number of classes: {num_classes}")

    print("\n" + "=" * 60)
    print("MODEL")
    print("=" * 60)

    # ---------------- BUILD MODEL ----------------
    conv_ch = config.get('conv_channels', 16)
    conv_k = config.get('conv_kernel', 3)
    flat_size = conv_ch * 15 * 15

    conv = Conv2D(3, conv_ch, conv_k)
    relu = ReLU()
    pool = MaxPool2D(2)
    flat = Flatten()
    fc = Linear(flat_size, num_classes)

    params = conv.parameters() + fc.parameters()

    # load weights
    load_model(params, args.weights_path)

    # disable gradients (evaluation mode)
    for p in params:
        p.requires_grad = False

    # ---------------- EVALUATION ----------------
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    correct = 0
    total = 0
    batch_count = 0

    for X_batch, y_batch in dataset.get_batches(args.batch_size):

        # ----- forward -----
        out = conv(X_batch)
        out = relu(out)
        out = pool(out)
        out = flat(out)
        out = fc(out)

        # predictions
        for i in range(len(y_batch)):
            pred = out.data[i].index(max(out.data[i]))
            if pred == y_batch[i]:
                correct += 1

        total += len(y_batch)
        batch_count += 1

        if batch_count % 10 == 0:
            print(f"Evaluated {total}/{len(dataset)} samples...")

    accuracy = correct / total if total > 0 else 0.0

    print("\nFinal Accuracy: {:.4f} ({}/{})".format(
        accuracy, correct, total
    ))


# ------------------------------------------------------------
if __name__ == '__main__':
    main()