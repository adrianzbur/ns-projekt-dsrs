import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation blok.

    Parametre
    ----------
    channels : int
        Počet vstupných/výstupných kanálov.
    reduction : int
        Faktor zmenšenia pre bottleneck (typicky 16).
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()

        bottleneck = max(1, channels // reduction)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # (B, C, H, W) → (B, C, 1, 1)
            nn.Flatten(),                      # → (B, C)
            nn.Linear(channels, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # scale: (B, C) → (B, C, 1, 1)
        scale = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class ConvBlock(nn.Module):
    """
    Jeden konvolučný blok: Conv2d → BN → ReLU → SEBlock → MaxPool.

    Parametre
    ----------
    in_ch  : vstupné kanály
    out_ch : výstupné kanály
    kernel : veľkosť konvolučného jadra (default 3×3)
    pool   : veľkosť MaxPool okna (default 2×2)
    se_reduction : SE reduction ratio
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 3,
        pool: int = 2,
        se_reduction: int = 16,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=kernel,
            padding=kernel // 2,   # same padding
            bias=False,
        )
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)
        self.se   = SEBlock(out_ch, reduction=se_reduction)
        self.pool = nn.MaxPool2d(pool)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class CNN(nn.Module):
    """
    Konvolučná sieť pre klasifikáciu stresu zo spektrogramu.

    Parametre
    ----------
    in_channels   : počet vstupných kanálov (default 1 — grayscale)
    base_channels : počet kanálov prvého bloku (ďalšie sa zdvojnásobia)
    n_blocks      : počet ConvBlock-ov (default 3 → 32→64→128)
    dropout       : dropout pred výstupnou vrstvou
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        n_blocks: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()

        # Konvolučné bloky: kanály 1 → 32 → 64 → 128
        blocks = []
        ch = in_channels
        for i in range(n_blocks):
            out_ch = base_channels * (2 ** i)   # 32, 64, 128
            blocks.append(ConvBlock(ch, out_ch))
            ch = out_ch

        self.features = nn.Sequential(*blocks)

        # Global Average Pooling — zredukuje (B, C, H, W) → (B, C)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.dropout = nn.Dropout(p=dropout)

        # Výstupný logit
        self.classifier = nn.Linear(ch, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 1, n_mels, time_frames)
        
        Vráti logit tvaru (B,).
        """
        # Konvolučné bloky
        x = self.features(x)            # (B, 128, H', W')

        # Global Average Pooling
        x = self.gap(x)                 # (B, 128, 1, 1)
        x = x.flatten(1)               # (B, 128)

        x = self.dropout(x)

        # Logit
        x = self.classifier(x)         # (B, 1)
        x = x.squeeze(-1)              # (B,)

        return x


if __name__ == "__main__":
    # Rýchly smoke test
    print("=== CNN smoke test ===")

    model = CNN()

    # Normálny vstup: (B, 1, 128, 94)
    x = torch.randn(8, 1, 128, 94)
    out = model(x)
    print(f"Vstup {tuple(x.shape)} → výstup {tuple(out.shape)}")
    assert out.shape == (8,), f"Očakávané (8,), dostali {out.shape}"

    # Skontroluj gradient flow
    loss = out.sum()
    loss.backward()
    print("Gradient flow: OK")

    # Počet parametrov
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Celkový počet parametrov : {total:,}")
    print(f"Trénovateľné parametre   : {trainable:,}")

    # Zobraz architektúru
    print("\n--- Architektúra ---")
    print(model)
    print("OK")