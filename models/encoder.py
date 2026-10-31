import torch
import torch.nn as nn
from torchvision import models


class EncoderCNN(nn.Module):
    """Pretrained ResNet-50 (frozen) -> 7x7 grid of features -> projected to embed_size.

    pretrained=True downloads the ImageNet weights (needs Internet). The default is False because
    CombinedModel(...) builds this class with defaults and a trained checkpoint is loaded right after.
    """

    def __init__(self, embed_size, dropout_p=0.3, pretrained=False):
        super(EncoderCNN, self).__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])  # drop avgpool + fc -> (B, 2048, 7, 7)
        for p in self.cnn.parameters():
            p.requires_grad = False  # backbone stays frozen
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(2048, embed_size)

    def train(self, mode=True):
        super(EncoderCNN, self).train(mode)
        self.cnn.eval()  # keep the frozen backbone's BatchNorm in eval mode, even during training
        return self

    def forward(self, images):
        # images: (B, 3, H, W)
        with torch.no_grad():
            features = self.cnn(images)  # (B, 2048, H', W')
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, num_pixels, 2048)
        return self.fc(self.dropout(features))  # (B, num_pixels, embed_size)
