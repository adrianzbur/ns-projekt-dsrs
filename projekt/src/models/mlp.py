import torch
import torch.nn as nn
 
 
class MLP(nn.Module):
    """
    Viacvrstvový perceptrón pre klasifikáciu stresu z akustických príznakov.
 
    Parametre
    ----------
    input_dim : int
        Rozmer vstupného vektora. Pre WS3D = 120, pre TESS = 13.
        Ak je vstup 2D (napr. MFCC matica tvaru [D, T]), model
        automaticky priemerne cez časovú os.
    hidden_dims : tuple[int]
        Rozmery skrytých vrstiev.
    dropout : float
        Pravdepodobnosť dropout-u (rovnaká pre všetky vrstvy).
    """
 
    def __init__(
        self,
        input_dim: int = 120,
        hidden_dims: tuple = (256, 128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
 
        self.input_dim = input_dim
 
        # Vstupná batch normalizácia
        self.input_bn = nn.BatchNorm1d(input_dim)
 
        # Skryté vrstvy
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_dim = h_dim
 
        self.hidden = nn.Sequential(*layers)
 
        # Výstupná vrstva — binárna klasifikácia, výstup = logit
        self.output = nn.Linear(in_dim, 1)
 
        # Inicializácia váh
        self._init_weights()
 
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, D) alebo (B, D, T)
            B = batch size
            D = feature dim
            T = time frames (ak je vstup 2D MFCC matica)
        
        Vráti logit tvaru (B,).
        """
        # Ak je vstup 3D (B, D, T) — priemer cez časovú os
        if x.dim() == 3:
            x = x.mean(dim=2)   # → (B, D)
 
        # Vstupná normalizácia
        x = self.input_bn(x)
 
        # Skryté vrstvy
        x = self.hidden(x)
 
        # Výstup — logit (B, 1) → (B,)
        x = self.output(x).squeeze(-1)
 
        return x
 
 
if __name__ == "__main__":
    # Rýchly smoke test
    print("=== MLP smoke test ===")
 
    # WS3D vstup: (B, 120) — flat feature vector
    model_ws3d = MLP(input_dim=120)
    x_ws3d = torch.randn(8, 120)
    out = model_ws3d(x_ws3d)
    print(f"WS3D flat  input {tuple(x_ws3d.shape)} → output {tuple(out.shape)}")
    assert out.shape == (8,), f"Očakávané (8,), dostali {out.shape}"
 
    # WS3D vstup: (B, 120, T) — MFCC matica
    x_ws3d_2d = torch.randn(8, 120, 94)
    out2 = model_ws3d(x_ws3d_2d)
    print(f"WS3D matrix input {tuple(x_ws3d_2d.shape)} → output {tuple(out2.shape)}")
    assert out2.shape == (8,)
 
    # TESS vstup: (B, 13, 94)
    model_tess = MLP(input_dim=13)
    x_tess = torch.randn(8, 13, 94)
    out3 = model_tess(x_tess)
    print(f"TESS matrix input {tuple(x_tess.shape)} → output {tuple(out3.shape)}")
    assert out3.shape == (8,)
 
    total_params = sum(p.numel() for p in model_ws3d.parameters())
    print(f"Počet parametrov (WS3D): {total_params:,}")
    print("OK")
