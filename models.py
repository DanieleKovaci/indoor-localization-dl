"""
models.py — MLP and 1D CNN for UJIIndoorLoc building/floor classification.

Both models return raw logits; use CrossEntropyLoss during training.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Multi-layer perceptron with BatchNorm and Dropout.

    Architecture (default):
        Linear(520→512) → BN → ReLU → Dropout
        Linear(512→256) → BN → ReLU → Dropout
        Linear(256→128) → BN → ReLU → Dropout
        Linear(128→num_classes)
    """

    def __init__(
        self,
        input_dim: int = 520,
        hidden_dims: list = None,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1D(nn.Module):
    """
    1D convolutional network that treats the 520-WAP vector as a 1D signal.

    Architecture:
        (batch, 520) → reshape → (batch, 1, 520)
        Conv(1→32, k=5, pad=2) → BN → ReLU                  length 520
        Conv(32→64, k=5, pad=2) → BN → ReLU → MaxPool(2)    length 260
        Conv(64→128, k=3, pad=1) → BN → ReLU → MaxPool(2)   length 130
        Flatten → Linear(128×130, 256) → ReLU → Dropout → Linear(256, num_classes)
    """

    def __init__(
        self,
        input_dim: int = 520,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Convolutional blocks
        self.conv_blocks = nn.Sequential(
            # Block 1 — no pooling, preserves length 520
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # Block 2 — pool → length 260
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            # Block 3 — pool → length 130
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        flat_dim = 128 * 130  # 16640

        self.classifier = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 520) → (batch, 1, 520)
        x = x.unsqueeze(1)
        x = self.conv_blocks(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_model(model_name: str, num_classes: int) -> nn.Module:
    """Factory function. model_name: 'mlp' or 'cnn'."""
    model_name = model_name.lower()
    if model_name == "mlp":
        return MLP(num_classes=num_classes)
    elif model_name == "cnn":
        return CNN1D(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choose 'mlp' or 'cnn'.")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    batch = 8
    x = torch.randn(batch, 520)

    for name, n_cls in [("mlp", 3), ("cnn", 5)]:
        model = build_model(name, n_cls)
        out = model(x)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"{name.upper():6s}  classes={n_cls}  "
            f"output shape: {tuple(out.shape)}  "
            f"params: {n_params:,}"
        )
        assert out.shape == (batch, n_cls), "Shape mismatch!"

    print("Smoke test passed.")
