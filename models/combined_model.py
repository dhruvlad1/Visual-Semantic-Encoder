import torch.nn as nn

from .decoder import DecoderWithAttention
from .encoder import EncoderCNN


class CombinedModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim=256, pretrained=False):
        super(CombinedModel, self).__init__()
        # pretrained=True downloads the ImageNet ResNet-50 weights; use it for training only.
        self.encoder = EncoderCNN(embed_size, pretrained=pretrained)
        self.decoder = DecoderWithAttention(embed_size, hidden_size, vocab_size, attention_dim)

    def forward(self, images, captions):
        encoder_out = self.encoder(images)  # (B, num_pixels, embed)
        outputs = self.decoder(encoder_out, captions)  # (B, T-1, vocab_size)
        return outputs
