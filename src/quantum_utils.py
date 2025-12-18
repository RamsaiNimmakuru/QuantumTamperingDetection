import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as pnp
import math
from typing import Optional

class QuantumLayer(nn.Module):
    """
    Quantum feature layer using:
      • AngleEmbedding
      • StronglyEntanglingLayers
      • Learnable classical projection (input_dim → n_qubits)
      • CPU-safe QNode with interface='torch'
      • Batch-safe forward
    """

    def __init__(self,
                 input_dim: int,
                 n_qubits: int = 8,
                 n_layers: int = 3,
                 dev_name: str = "default.qubit",
                 shots: Optional[int] = None):
        super().__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots

        # Classical learnable projection
        self.proj = nn.Linear(input_dim, n_qubits)

        # Learnable scale & bias for normalization before embedding
        self.scale = nn.Parameter(torch.ones(n_qubits) * 1.0)
        self.bias = nn.Parameter(torch.zeros(n_qubits))

        # Pennylane device (CPU only)
        self.dev = qml.device(dev_name, wires=n_qubits, shots=shots)

        # QNode definition
        weight_shape = (n_layers, n_qubits, 3)

        @qml.qnode(self.dev, interface='torch', diff_method='backprop')
        def qnode(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = qnode
        self.q_weights = nn.Parameter(0.01 * torch.randn(weight_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, input_dim)
        Output: (batch, n_qubits)
        """
        if x.ndim != 2:
            raise ValueError("QuantumLayer expects shape (batch_size, input_dim)")

        batch_size = x.shape[0]

        # Classical projection → (B, n_qubits)
        proj = self.proj(x)

        # Normalize to [0, π]
        proj = torch.sigmoid(proj * self.scale + self.bias) * math.pi

        outputs = []
        cpu = torch.device("cpu")
        orig_device = proj.device

        qweights_cpu = self.q_weights.to(cpu)

        for i in range(batch_size):
            inp = proj[i].to(cpu)
            out = self.qnode(inp, qweights_cpu)

            # guarantee torch tensor
            if not isinstance(out, torch.Tensor):
                out = torch.tensor(out, dtype=torch.float32, device=cpu)

            outputs.append(out)

        return torch.stack(outputs).to(orig_device)

