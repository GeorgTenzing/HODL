import torch
import torch.nn as nn
import torch.nn.functional as F

# list_of_blocks = [Conv1d_block, SeparableConv1d, in_block, Deconv, Bottleneck, WaveUNetBlock, DCUNetBlock, TinyConvBlock, ResidualBlock, AttentionBlock, GatedConv1d]

class Conv1d_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=5, stride=1, padding=2): 
        super().__init__()
        self.conv     = nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)
        #self.norm     = nn.BatchNorm1d(out_c)
        self.activate = nn.LeakyReLU(0.1, inplace=True)      # nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.activate(self.conv(x))
        #return self.activate(self.norm(self.conv(x)))
        
        
        
        

# --- Separable Conv (Depthwise + Pointwise) ---
class SeparableConv1d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1): 
        super().__init__()
        self.depthwise = nn.Conv1d(in_c, in_c, kernel_size, padding=padding, groups=in_c)
        self.pointwise = nn.Conv1d(in_c, out_c, kernel_size=1)
        # self.activate =  nn.LeakyReLU()

    def forward(self, x):
        return F.relu(self.pointwise(self.depthwise(x)))
        # return self.activate(self.pointwise(self.depthwise(x)))
    
    
    
# NEW BLOCKS

class  in_block(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=7, stride=3, padding=1): 
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.conv  = nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        return x + (1.0/self.alpha) * torch.pow(torch.sin(self.alpha * x), 2)
        
class Deconv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=2, stride=2): 
        super().__init__()
        self.conv     = nn.ConvTranspose1d(in_c, out_c, kernel_size=kernel_size, stride=stride)
        self.activate = nn.LeakyReLU()

    def forward(self, x):
        return self.activate(self.conv(x))




class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=5, stride=1, padding=2): 
        super().__init__()
        self.conv     = nn.Conv1d(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)

    
    
# =====================================================================
# === New Block Definitions (from GitHub repos, adapted to your format)
# =====================================================================

class WaveUNetBlock(nn.Module):
    """Residual-style conv block with large receptive field (Wave-U-Net inspired)."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=15, padding=7)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=15, padding=7)
        self.norm = nn.BatchNorm1d(out_c)
        self.act = nn.LeakyReLU(0.2)

        self.proj = nn.Conv1d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        skip = self.proj(x)
        x = self.act(self.conv1(x))
        x = self.act(self.norm(self.conv2(x)))
        return x + skip


class DCUNetBlock(nn.Module):
    """Dual-path gated block approximating DCUNet (magnitude + phase pathway)."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.mag_path = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.PReLU()
        )
        self.phase_path = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.Sigmoid()
        )

        self.res_proj = nn.Conv1d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        mag = self.mag_path(x)
        phase = self.phase_path(x)
        return mag * phase + self.res_proj(x)


class TinyConvBlock(nn.Module):
    """Lightweight convolutional block from Noise2Noise (fast for leaderboard training)."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.1),
            nn.Conv1d(out_c, out_c, kernel_size=5, padding=2),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.net(x)
    
    
    
# =====================================================================
# === Blocks that have not shown any increase
# =====================================================================   
    
# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.proj = nn.Conv1d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()
        self.block = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + self.proj(x))
        
# Attention Block (Squeeze-and-Excitation style)
class AttentionBlock(nn.Module):
    def __init__(self, in_c, out_c, reduction=8):
        super().__init__()
        channels = in_c
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1)
        return x * w
        
# --- Gated Convolution ---
class GatedConv1d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super().__init__()
        self.feature_conv = nn.Conv1d(in_c, out_c, kernel_size, padding=padding)
        self.gate_conv    = nn.Conv1d(in_c, out_c, kernel_size, padding=padding)

    def forward(self, x):
        return torch.tanh(self.feature_conv(x)) * torch.sigmoid(self.gate_conv(x))

