import torch
from torch import nn


class MaxFeatureMap(nn.Module):
    """Pairwise channel-wise maximum used by Light CNN."""

    def forward(self, inputs):
        if inputs.shape[1] % 2:
            raise ValueError("MFM expects an even number of channels/features")
        first, second = inputs.chunk(2, dim=1)
        return torch.maximum(first, second)


def _conv_mfm(in_channels, out_channels, kernel_size, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels * 2, kernel_size, padding=padding),
        MaxFeatureMap(),
    )


class LCNN(nn.Module):
    """LCNN classifier reconstructed from the architecture table in STC 2019."""

    def __init__(
        self,
        input_frequency_bins=257,
        input_frames=750,
        dropout=0.75,
        num_classes=2,
    ):
        super().__init__()
        self.features = nn.Sequential(
            _conv_mfm(1, 32, 5, 2),
            nn.MaxPool2d(2),
            _conv_mfm(32, 32, 1, 0),
            nn.BatchNorm2d(32),
            _conv_mfm(32, 48, 3, 1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(48),
            _conv_mfm(48, 48, 1, 0),
            nn.BatchNorm2d(48),
            _conv_mfm(48, 64, 3, 1),
            nn.MaxPool2d(2),
            _conv_mfm(64, 64, 1, 0),
            nn.BatchNorm2d(64),
            _conv_mfm(64, 32, 3, 1),
            nn.BatchNorm2d(32),
            _conv_mfm(32, 32, 1, 0),
            nn.BatchNorm2d(32),
            _conv_mfm(32, 32, 3, 1),
            nn.MaxPool2d(2),
        )
        self.features.eval()
        with torch.no_grad():
            shape_probe = torch.zeros(1, 1, input_frequency_bins, input_frames)
            flattened_size = self.features(shape_probe).numel()
        self.features.train()
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, 160),
            MaxFeatureMap(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(80),
            nn.Linear(80, num_classes),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, data_object, **batch):
        features = self.features(data_object).flatten(start_dim=1)
        return {"logits": self.classifier(features)}

    def __str__(self):
        parameters = sum(parameter.numel() for parameter in self.parameters())
        return f"{super().__str__()}\nAll parameters: {parameters}"
