import torch.nn as nn

from malaria.model import BaseLitModel


class MalariaLitModel(BaseLitModel):
    """
    PyTorch Lightning Module for malaria cell classification using a simple CNN.
    Handles model definition, training, validation, and test steps.
    """

    def __init__(self, num_classes=2, lr=1e-3):
        super().__init__(num_classes=num_classes, lr=lr)
        # CNN architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x, *args, **kwargs):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output tensor after classification layer (e.g., logits or class scores).
                - Embedding tensor representing the feature vector before classification.
        """

        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        embedding = x
        x = self.classifier(x)
        return x, embedding
