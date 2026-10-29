import torch
from torch import nn


def conv_block(c_in, c_out):
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CopperCNN(nn.Module):
    """Small VGG-style CNN with two outputs: a logit for Cu >= threshold and predicted log10 Cu."""

    def __init__(self, in_channels=4, widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        chans = (in_channels,) + tuple(widths)
        self.body = nn.Sequential(
            *[conv_block(a, b) for a, b in zip(chans[:-1], chans[1:])],
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(widths[-1], 64),
            nn.ReLU(inplace=True),
        )
        self.bin_head = nn.Linear(64, 1)
        self.log_head = nn.Linear(64, 1)

    def forward(self, x):
        features = self.body(x)
        return self.bin_head(features).squeeze(1), self.log_head(features).squeeze(1)


if __name__ == "__main__":
    model = CopperCNN()
    print(f"{sum(p.numel() for p in model.parameters()):,} parameters")
    print([t.shape for t in model(torch.zeros(2, 4, 32, 32))])
