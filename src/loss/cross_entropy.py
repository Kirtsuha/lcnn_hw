from torch import nn


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels, **batch):
        return {"loss": self.criterion(logits, labels)}
