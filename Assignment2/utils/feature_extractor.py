import torch

def extract_features(model, layer_name, loader, device):

    features = []
    labels = []

    def hook(module, input, output):
        feats = output.mean(dim=[2,3]) if output.dim()==4 else output
        features.append(feats.detach().cpu())

    layer = dict(model.named_modules())[layer_name]

    handle = layer.register_forward_hook(hook)

    model.eval()

    with torch.no_grad():
        for x,y in loader:

            x = x.to(device)
            model(x)

            labels.append(y)

    handle.remove()

    features = torch.cat(features)
    labels = torch.cat(labels)

    return features, labels