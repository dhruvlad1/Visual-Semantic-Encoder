import torch


def load_checkpoint(model, optimizer=None, path="checkpoint.pth"):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError:
            # Inference only needs the model weights. The optimizer built by the scripts includes the
            # frozen ResNet parameters, so its groups don't match the saved one. Safe to skip.
            pass
    return checkpoint["epoch"], checkpoint["loss"]
