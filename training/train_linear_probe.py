import torch
import torch.nn as nn
import torch.optim as optim

def train_probe(features, labels, num_classes):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    clf = nn.Linear(features.shape[1], num_classes).to(device)

    optimizer = optim.Adam(clf.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(features, labels)

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(10):

        for x,y in loader:

            x = x.to(device)
            y = y.to(device)

            out = clf(x)

            loss = criterion(out,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return clf

