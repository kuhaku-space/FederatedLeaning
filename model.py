import torch
import torch.nn as nn


# --- モデル定義 ---
class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            # Conv1: filters=32, kernel_size=[5, 5], padding="same"
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            # Pool1: pool_size=[2, 2], strides=2
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv2: filters=64, kernel_size=[5, 5], padding="same"
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            # Pool2: pool_size=[2, 2], strides=2
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Dense層
        # 入力次元: 7 * 7 * 64 = 3136
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Dense1: units=248, activation=relu
            nn.Linear(3136, 248),
            nn.ReLU(),
            # Logits: units=num_classes
            nn.Linear(248, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
