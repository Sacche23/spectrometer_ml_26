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

@register_model("wen_mlp") #aka: dnn
class SpectrumDNN(nn.Module):
    """
    MLP adapted from Wen et al. (ACS Photonics, 2023)
    Input-BN-LR-FC-500-BN-LR-FC-500-BN-LR-FC-301-BN-LR-Output
    """ 
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 500)
        self.bn1 = nn.BatchNorm1d(500)

        self.fc2 = nn.Linear(500, 500)
        self.bn2 = nn.BatchNorm1d(500)

        self.fc3 = nn.Linear(500, output_dim)
        # self.bn3 = nn.BatchNorm1d(500)

        # self.fc4 = nn.Linear(500, output_dim)
        
        # Leaky ReLU activation.
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        # ReLU activation
        self.relu = nn.ReLU()

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
        # x = self.bn3(x)
        # x = self.leaky_relu(x)
        # x = self.dropout(x)

        # Output layer: FC → BN → LeakyReLU
        # No BN, LR, or dropout on the output layer.
        # x = self.fc4(x)          # → (batch_size, output_dim)
        x = self.relu(x)

        # RMS Normalization
        ms = torch.mean(x.pow(2), dim=1, keepdim=True)
        rms = torch.sqrt(ms + 1e-6)
        x = x / rms

        return x

@register_model("unet")
class UNet1DBhatti(nn.Module):
    """
    1D U-Net following Bhatti et al. (2025).
    Dimensions toned down to be more lightweight

    Key features vs. a plain U-Net:
    1. Dense pre-expansion: maps input to output_dim before U-Net
    2. Residual blocks in contracting path (ResNet-style shortcut within each block)
    3. Strided Conv for downsampling (not MaxPool)
    4. Standard ConvTranspose + concatenate + residual block in expansive path
    5. Global residual: U-Net output + pre-expanded input
    * Note: Paper states a 50% dropout rate but not where in the architecture, assumed to be at the bottleneck
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        n_channels = 8
        dropout_rate = 0.2

        # --- Step 1: Dense layer pre-expansion ---
        # Maps 36 (or N) intensities → output_dim (350/1000/823)
        # This is the linear transformation that "extends" the measurement
        # vector to the full output resolution before the U-Net sees it.
        self.dense = nn.Linear(input_dim, output_dim)

        # --- Contracting path (encoder) ---
        # Stage 1: no downsampling, establishes base feature maps
        self.enc1 = self._residual_block(1, n_channels, stride=1)

        # Stages 2-5: each halves the sequence length and doubles channels
        # stride=2 in the Conv replaces MaxPool
        self.enc2 = self._residual_block(n_channels, n_channels*2, stride=2)
        self.enc3 = self._residual_block(n_channels*2, n_channels*4, stride=2)
        self.enc4 = self._residual_block(n_channels*4, n_channels*8, stride=2)
        self.enc5 = self._residual_block(n_channels*8, n_channels*16, stride=2)  # bottleneck
        self.dropout = nn.Dropout1d(p=dropout_rate) # 0.5 in the paper

        # --- Expansive path (decoder) ---
        # Each stage: upsample → concatenate skip → residual block
        self.up4 = nn.ConvTranspose1d(n_channels*16, n_channels*8, kernel_size=2, stride=2)
        self.dec4 = self._residual_block(n_channels*16, n_channels*8, stride=1)  # 128 due to concat

        self.up3 = nn.ConvTranspose1d(n_channels*8, n_channels*4, kernel_size=2, stride=2)
        self.dec3 = self._residual_block(n_channels*8, n_channels*4, stride=1)

        self.up2 = nn.ConvTranspose1d(n_channels*4, n_channels*2, kernel_size=2, stride=2)
        self.dec2 = self._residual_block(n_channels*4, n_channels*2, stride=1)

        self.up1 = nn.ConvTranspose1d(n_channels*2, n_channels, kernel_size=2, stride=2)
        self.dec1 = self._residual_block(n_channels*2, n_channels, stride=1)

        # Final conv: maps 8 feature maps back to 1 channel (the spectrum)
        self.final_conv = nn.Conv1d(n_channels, 1, kernel_size=1)

    def _residual_block(self, in_channels, out_channels, stride=1):
        """
        One residual block as described by Bhatti et al.:

        Main branch:   Conv(stride) → BaN → ReLU → Conv → BaN
        Shortcut:      Conv(stride) → BaN
        Output:        main + shortcut (then ReLU applied after addition)

        If stride=2, this block also halves the sequence length.
        The shortcut uses a 1x1 Conv to match dimensions.
        """
        return ResidualBlock1D(in_channels, out_channels, stride)
    
    def _pad_to_match(self, x, target):
        diff = target.shape[-1] - x.shape[-1]
        if diff > 0:
            x = F.pad(x, (0, diff))
        elif diff < 0:
            # x is longer → crop the extra values
            x = x[..., :target.shape[-1]]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, input_dim]

        # Step 1: Dense pre-expansion
        # "36 measured intensities were extended to a size of 350 by
        #  applying a linear transformation using the dense layer"
        x_expanded = self.dense(x)          # [batch, output_dim]

        # Reshape for 1D convolutions: add channel dimension
        # [batch, output_dim] → [batch, 1, output_dim]
        h = x_expanded.unsqueeze(1)

        # Step 2: Contracting path (save each stage for skip connections)
        e1 = self.enc1(h)   # [batch, 8,   L]
        e2 = self.enc2(e1)  # [batch, 16,  L/2]
        e3 = self.enc3(e2)  # [batch, 32,  L/4]
        e4 = self.enc4(e3)  # [batch, 64,  L/8]
        e5 = self.enc5(e4)  # [batch, 128, L/16] — bottleneck
        e5 = self.dropout(e5) # Dropout 20%

        # Step 3: Expansive path with skip connections
        # After each upsample, concatenate with corresponding encoder output,
        # then pass through residual block (repeated 4 times as stated)
        d4 = self.up4(e5)
        d4 = self._pad_to_match(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self._pad_to_match(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self._pad_to_match(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self._pad_to_match(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        # Final conv: [batch, 8, L] → [batch, 1, L] → [batch, L]
        unet_out = self.final_conv(d1).squeeze(1)
        
        # Step 4: Global residual
        # "the output signal and extended intensities were added"
        # This means the U-Net only needs to learn the correction,
        # not the full spectrum from scratch
        out = unet_out + x_expanded

        # RMS Normalization
        ms = torch.mean(out.pow(2), dim=1, keepdim=True)
        rms = torch.sqrt(ms + 1e-6)
        out = out / rms

        return out  
class ResidualBlock1D(nn.Module):
    """
    The core building block. Two branches that are summed:

    Main:     Conv(k=3, stride) → BaN → ReLU → Conv(k=3) → BaN
    Shortcut: Conv(k=1, stride) → BaN   ← matches dimensions

    The two branches are summed, followed by a ReLU activation.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Main branch
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels,
                      kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(out_channels),
        )

        # Shortcut branch — only needed when dimensions change
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels,
                          kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.main(x) + self.shortcut(x))

@register_model("cui_mlp")
class SpectrumDNN(nn.Module):
    """
    # MLP adapted from Cui et al. (Optics Communications, 2026)
    Original paper: MLP refines a MAP solver output (FMAP) → final spectrum.
    Adaptation: applied directly to photocurrent vectors, skipping MAP pre-processing.

    Architecture: Input → FC(256) → GELU → FC(256) → GELU → FC(output) → Softplus
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 256)
        # self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 256)
        # self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, output_dim)
        
        # GELU activation
        self.gelu = nn.GELU()
        # Softplus activation
        self.softplus = nn.Softplus()

        # Dropout for regularization
        # p=0.3 means 30% of neurons are dropped each step.
        # self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # x: input tensor of shape (batch_size, input_dim)
        #    e.g., (128, 1000) for a batch of 128 photocurrent vectors
        # returns: tensor of shape (batch_size, output_dim)
        
        # Layer 1: FC → BN → GELU
        x = self.fc1(x)
        # x = self.bn1(x)
        x = self.gelu(x)
        # x = self.dropout(x)

        # Layer 2: FC → BN → GELU
        x = self.fc2(x)
        # x = self.bn2(x)
        x = self.gelu(x)
        # x = self.dropout(x)

        # Output layer: Softplus
        # No BN, GL, or dropout on the output layer.
        x = self.fc3(x)          # → (batch_size, output_dim)
        x = self.softplus(x)

        # RMS Normalization
        ms = torch.mean(x.pow(2), dim=1, keepdim=True)
        rms = torch.sqrt(ms + 1e-6)
        x = x / rms

        return x
    
@register_model("cui_mlp_v2")
class SpectrumDNN(nn.Module):
    """
    # MLP adapted from Cui et al. (Optics Communications, 2026)
    Original paper: MLP refines a MAP solver output (FMAP) → final spectrum.
    Adaptation: applied directly to photocurrent vectors, skipping MAP pre-processing.
    V2: Added dropout, BN, for better performance.

    Architecture: 
    Input → FC(256) → BN → GELU → FC(256) → BN → GELU → FC(output) → Softplus
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)

        self.fc3 = nn.Linear(1024, output_dim)
        
        # GELU activation
        self.gelu = nn.GELU()
        # Softplus activation
        self.softplus = nn.Softplus()

        # Dropout for regularization
        # p=0.3 means 30% of neurons are dropped each step.
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        # x: input tensor of shape (batch_size, input_dim)
        #    e.g., (128, 1000) for a batch of 128 photocurrent vectors
        # returns: tensor of shape (batch_size, output_dim)
        
        # Layer 1: FC → BN → GELU
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.gelu(x)
        x = self.dropout(x)

        # Layer 2: FC → BN → GELU
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.gelu(x)
        x = self.dropout(x)

        # Output layer: Softplus
        # No BN, GL, or dropout on the output layer.
        x = self.fc3(x)          # → (batch_size, output_dim)
        x = self.softplus(x)

        # RMS Normalization
        ms = torch.mean(x.pow(2), dim=1, keepdim=True)
        rms = torch.sqrt(ms + 1e-6)
        x = x / rms

        return x

def make_cui_mlp_v2(hidden1: int, hidden2: int):
    """
    Factory function: returns an MLP based on Cui et al. (2026) 
    with the given hidden layer dimensions.
    """
    class CuiMLPv2(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden1)
            self.bn1 = nn.BatchNorm1d(hidden1)

            self.fc2 = nn.Linear(hidden1, hidden2)
            self.bn2 = nn.BatchNorm1d(hidden2)

            self.fc3 = nn.Linear(hidden2, output_dim)

            self.gelu = nn.GELU()
            self.softplus = nn.Softplus()
            self.dropout = nn.Dropout(p=0.2)

        def forward(self, x):
            x = self.dropout(self.gelu(self.bn1(self.fc1(x))))
            x = self.dropout(self.gelu(self.bn2(self.fc2(x))))
            x = self.softplus(self.fc3(x))

            ms = torch.mean(x.pow(2), dim=1, keepdim=True)
            rms = torch.sqrt(ms + 1e-6)
            return x / rms

    return CuiMLPv2

# Now register one variant per (hidden1, hidden2) combination:
for h1 in [256, 512, 1024]:
    for h2 in [256, 512, 1024]:
        name = f"cui_mlp_v2_{h1}_{h2}"
        register_model(name)(make_cui_mlp_v2(h1, h2))

def make_unet(n_channels: int, dropout_rate: float):
    """
    Factory function: returns a Unet based on Bhatti et al. (2023)
    with the given initial number of channels and chokepoint dropout rate.
    """
    class BhattiUnet(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.output_dim = output_dim
    
            # --- Step 1: Dense layer pre-expansion ---
            self.dense = nn.Linear(input_dim, output_dim)
    
            # --- Contracting path (encoder) ---
            self.enc1 = self._residual_block(1, n_channels, stride=1)
    
            # Stages 2-5: 
            self.enc2 = self._residual_block(n_channels, n_channels*2, stride=2)
            self.enc3 = self._residual_block(n_channels*2, n_channels*4, stride=2)
            self.enc4 = self._residual_block(n_channels*4, n_channels*8, stride=2)
            self.enc5 = self._residual_block(n_channels*8, n_channels*16, stride=2)  # bottleneck
            self.dropout = nn.Dropout1d(p=dropout_rate) # 0.5 in the paper
    
            # --- Expansive path (decoder) ---
            self.up4 = nn.ConvTranspose1d(n_channels*16, n_channels*8, kernel_size=2, stride=2)
            self.dec4 = self._residual_block(n_channels*16, n_channels*8, stride=1)  # 128 due to concat
    
            self.up3 = nn.ConvTranspose1d(n_channels*8, n_channels*4, kernel_size=2, stride=2)
            self.dec3 = self._residual_block(n_channels*8, n_channels*4, stride=1)
    
            self.up2 = nn.ConvTranspose1d(n_channels*4, n_channels*2, kernel_size=2, stride=2)
            self.dec2 = self._residual_block(n_channels*4, n_channels*2, stride=1)
    
            self.up1 = nn.ConvTranspose1d(n_channels*2, n_channels, kernel_size=2, stride=2)
            self.dec1 = self._residual_block(n_channels*2, n_channels, stride=1)
    
            # Final conv: 
            self.final_conv = nn.Conv1d(n_channels, 1, kernel_size=1)
    
        def _residual_block(self, in_channels, out_channels, stride=1):
            return ResidualBlock1D(in_channels, out_channels, stride)
        
        def _pad_to_match(self, x, target):
            diff = target.shape[-1] - x.shape[-1]
            if diff > 0:
                x = F.pad(x, (0, diff))
            elif diff < 0:
                x = x[..., :target.shape[-1]]
            return x
    
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [batch, input_dim]
    
            # Step 1: Dense pre-expansion
            x_expanded = self.dense(x)      # [batch, output_dim]
    
            # Reshape for 1D convolutions: add channel dimension
            h = x_expanded.unsqueeze(1)     # [batch, output_dim] → [batch, 1, output_dim]

    
            # Step 2: Contracting path (save each stage for skip connections)
            e1 = self.enc1(h)       # [batch, n_ch,   L]
            e2 = self.enc2(e1)      # [batch, *2 ,  L/2]
            e3 = self.enc3(e2)      # [batch, *2 ,  L/4]
            e4 = self.enc4(e3)      # [batch, *2 ,  L/8]
            e5 = self.enc5(e4)      # [batch, *2 , L/16] — bottleneck
            e5 = self.dropout(e5)   # Bottleneck dropout
    
            # Step 3: Expansive path with skip connections
            d4 = self.up4(e5)
            d4 = self._pad_to_match(d4, e4)
            d4 = self.dec4(torch.cat([d4, e4], dim=1))
    
            d3 = self.up3(d4)
            d3 = self._pad_to_match(d3, e3)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
    
            d2 = self.up2(d3)
            d2 = self._pad_to_match(d2, e2)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
    
            d1 = self.up1(d2)
            d1 = self._pad_to_match(d1, e1)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
    
            # Final conv: [batch, n_ch , L] → [batch, 1, L] → [batch, L]
            unet_out = self.final_conv(d1).squeeze(1)

            # Step 4: Global residual
            # "the output signal and extended intensities were added"
            # This means the U-Net only needs to learn the correction,
            # not the full spectrum from scratch
            out = unet_out + x_expanded
    
            # RMS Normalization
            ms = torch.mean(out.pow(2), dim=1, keepdim=True)
            rms = torch.sqrt(ms + 1e-6)
            out = out / rms
    
            return out    
    return BhattiUnet

# Now register one variant per (n_channel, dropout rate) combination:
for n_ch in [8, 16, 32]:
    for dr in [0.2, 0.35, 0.5]:
        name = f"unet_{n_ch}_{dr}"
        register_model(name)(make_unet(n_ch, dr))

