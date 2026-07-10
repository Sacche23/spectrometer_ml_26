import torch
import torch.nn as nn
import torch.nn.functional as F

# ====================================================================================
# Registry
# ====================================================================================

MODEL_REGISTRY = {}

def register_model(name):
    '''
    Call "@register_model(name)" above
    class declaration to register name as a valid
    model parameter.
    '''
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model(name):
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")

# ====================================================================================
# Model Classes
# ====================================================================================

# Original network
@register_model("nn")
class SpectrometerNet(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super(SpectrometerNet, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# V1 CNN
@register_model("cnn1")
class SpectrometerCNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(64)
        self.pool  = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(0.3)

        reduced_length = input_dim // 4
        self.fc1 = nn.Linear(64 * reduced_length, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        # normalize
        ms = torch.mean(x.pow(2), dim=1, keepdim=True)
        rms = torch.sqrt(ms + 1e-6)
        x = x / rms

        return x

# Lower dim network, computes output as sum of evenly spaced gaussians
@register_model("gcnn")
class SpectrometerCNNGaussian(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 n_gaussians: int = 41):
        super().__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(64)
        self.pool  = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(0.3)
        reduced_length = input_dim // 4
        self.fc1 = nn.Linear(64 * reduced_length, 128)
        self.fc2 = nn.Linear(128, n_gaussians)
        centers = torch.linspace(0., 1., n_gaussians)
        pts = torch.linspace(0., 1., output_dim).unsqueeze(0)
        # choose a stdev so neighboring gaussians overlap smoothly
        sigma = 1.0 / (n_gaussians - 1)
        basis = torch.exp(-0.5 * ((pts - centers.unsqueeze(1)) / sigma)**2)
        basis = basis / basis.sum(dim=1, keepdim=True)
        self.register_buffer('basis', basis)  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        w = self.fc2(x)
        w = F.relu(w)
        w = w / (w.sum(dim=1, keepdim=True) + 1e-6)
        y = w @ self.basis
        ms = torch.mean(y.pow(2), dim=1, keepdim=True)
        rms = torch.sqrt(ms + 1e-6)
        y = y / rms

        return y

# V2 CNN
@register_model("cnn2")
class CNN2(nn.Module):
    def __init__(self, input_dim=41, output_dim=1000):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(64)
        self.pool  = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(0.3)

        reduced_length = input_dim // 4
        flat_features = 64 * reduced_length

        self.fc1 = nn.Linear(flat_features, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept either (B, input_dim) or (B, 1, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = x.flatten(1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        # normalize each sample to unit RMS
        ms  = torch.mean(x.pow(2), dim=1, keepdim=True)  # (B,1)
        rms = torch.sqrt(ms + 1e-6)
        x   = x / rms
        return x
    
#------------------------------------------------------------------
# Summer 2026
#------------------------------------------------------------------

@register_model("dnn")
class SpectrumDNN(nn.Module):
# MLP adapted from Wen et al. (ACS Photonics, 2023)
# Input-BN-LR-FC-500-BN-LR-FC-500-BN-LR-FC-301-BN-LR-Output
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 500)
        self.bn1 = nn.BatchNorm1d(500)

        self.fc2 = nn.Linear(500, 500)
        self.bn2 = nn.BatchNorm1d(500)

        self.fc3 = nn.Linear(500, 500)
        self.bn3 = nn.BatchNorm1d(500)

        self.fc4 = nn.Linear(500, output_dim)
        
        # Leaky ReLU activation.
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        # Dropout for regularization
        # p=0.3 means 30% of neurons are dropped each step.
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # x: input tensor of shape (batch_size, input_dim)
        #    e.g., (128, 1000) for a batch of 128 photocurrent vectors
        # returns: tensor of shape (batch_size, output_dim)
        
        # Layer 1: FC → BN → LeakyReLU
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        # Layer 2: FC → BN → LeakyReLU
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        
        # Layer 3: FC → BN → LeakyReLU
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        # Output layer: FC → BN → LeakyReLU
        # No BN, LR, or dropout on the output layer.
        x = self.fc4(x)          # → (batch_size, output_dim)

        # RMS Normalization
        ms = torch.mean(x.pow(2), dim=1, keepdim=True)
        rms = torch.sqrt(ms + 1e-6)
        x = x / rms

        return x