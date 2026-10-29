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
    """Small VGG-style CNN with two outputs: a logit for Cu >= threshold and predicted log10 Cu.

    rock_sizes: number of rock codes per categorical layer; each layer gets a learned
                rock_dim-channel embedding concatenated with the continuous channels.
    n_extra:    per-cell scalars (e.g. hole depth) concatenated after global pooling.
    """

    def __init__(self, in_channels=4, rock_sizes=(), rock_dim=4, n_extra=0,
                 widths=(32, 64, 128, 256), dropout=0.3):
        super().__init__()
        self.rock_emb = nn.ModuleList([nn.Embedding(n + 1, rock_dim) for n in rock_sizes])
        chans = (in_channels + rock_dim * len(rock_sizes),) + tuple(widths)
        self.features = nn.Sequential(
            *[conv_block(a, b) for a, b in zip(chans[:-1], chans[1:])],
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
        )
        self.fc = nn.Sequential(nn.Linear(widths[-1] + n_extra, 64), nn.ReLU(inplace=True))
        self.bin_head = nn.Linear(64, 1)
        self.log_head = nn.Linear(64, 1)

    def forward(self, x, rock=None, extra=None):
        if len(self.rock_emb):
            embedded = [emb(rock[:, i]).permute(0, 3, 1, 2) for i, emb in enumerate(self.rock_emb)]
            x = torch.cat([x] + embedded, dim=1)
        features = self.features(x)
        if extra is not None:
            features = torch.cat([features, extra], dim=1)
        features = self.fc(features)
        return self.bin_head(features).squeeze(1), self.log_head(features).squeeze(1)


if __name__ == "__main__":
    for kwargs in [{}, {"rock_sizes": (9, 2, 4, 19), "n_extra": 1}]:
        model = CopperCNN(**kwargs)
        rock = torch.zeros(2, 4, 64, 64, dtype=torch.long) if kwargs else None
        extra = torch.zeros(2, 1) if kwargs else None
        out = model(torch.zeros(2, 4, 64, 64), rock, extra)
        print(kwargs, f"{sum(p.numel() for p in model.parameters()):,} parameters", [t.shape for t in out])
