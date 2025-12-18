import torch
import torch.nn as nn
from torchvision import models
from src.quantum_utils import QuantumLayer

class HybridQuantumCNN(nn.Module):
    """
    Hybrid model:
      - CNN backbone (VGG16 features)
      - small learned classical projection to compact vector
      - QuantumLayer consuming compact vector -> n_qubits features
      - small classifier on top
    """
    def __init__(self, n_qubits: int = 8, pretrained_backbone: bool = True):
        super().__init__()
        # Backbone: VGG16 features
        if pretrained_backbone:
            backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        else:
            backbone = models.vgg16(weights=None)
        self.feature_extractor = backbone.features  # conv blocks
        # Adaptive pool to get fixed-size vector
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # Determine the feature dimension after pooling:
        # Usually VGG16 last conv channel = 512
        self.feature_dim = 512

        # Classical projection: map feature_dim -> projection_dim suitable for QuantumLayer
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        projection_output_dim = 128

        # Quantum layer (uses src/quantum_utils.QuantumLayer)
        self.quantum_layer = QuantumLayer(input_dim=projection_output_dim, n_qubits=n_qubits, n_layers=3)

        # Classifier on top of quantum outputs
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.feature_extractor(x)            # -> [B, C, h, w]
        x = self.pool(x)                         # -> [B, C, 1, 1]
        x = x.view(x.size(0), -1)                # -> [B, C]
        x = self.projection(x)                   # -> [B, projection_output_dim]
        q = self.quantum_layer(x)                # -> [B, n_qubits]
        out = self.classifier(q)                 # -> [B, 2]
        return out
