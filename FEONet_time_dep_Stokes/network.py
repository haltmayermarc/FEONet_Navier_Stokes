import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



#######################################################################
### UNet 2D Model
#######################################################################

# -----------------------------
# Building Blocks
# -----------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3, 1),
            ConvBNAct(out_ch, out_ch, 3, 1),
        )
    def forward(self, x):
        return self.block(x)

# -----------------------------
# UNet-like Feature Extractor
# -----------------------------
class UNetFeatureExtractor(nn.Module):
    """
    Input:  (B, 2, H, W)
    Output: (B, d, H, W) where d = latent_ch
    Preserves spatial size with strided pooling+upsampling.
    """
    def __init__(self, in_ch=2, base_ch=32, latent_ch=16):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_ch, base_ch)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(base_ch * 2, base_ch * 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        # Project to latent channels d
        self.proj = nn.Conv2d(base_ch, latent_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #B, C, H, W = x.shape
        # Encoder
        e1 = self.enc1(x)              # (B, base_ch, H, W)
        e2 = self.enc2(self.pool1(e1)) # (B, 2*base_ch, H/2, W/2)

        # Bottleneck
        b = self.bottleneck(self.pool2(e2)) # (B, 4*base_ch, H/4, W/4)

        # Decoder with skips
        d2 = self.up2(b)                       # (B, 2*base_ch, H/2, W/2)
        d2 = self.dec2(torch.cat([d2, e2], 1)) # (B, 2*base_ch, H/2, W/2)

        d1 = self.up1(d2)                      # (B, base_ch, H, W)
        d1 = self.dec1(torch.cat([d1, e1], 1)) # (B, base_ch, H, W)

        latent = self.proj(d1)                 # (B, latent_ch, H, W)
        return latent

# -----------------------------
# Prediction Head (like Net2D)
# -----------------------------
class UNetHead(nn.Module):
    """
    Input:  (B, d, H, W)
    Output: (B, seq_len, d_out)
    """
    def __init__(self, d_in: int, d_out: int, hidden: int = 128):
        super().__init__()
        self.act = nn.SiLU(inplace=True)

        # compress spatial features into a global vector
        self.pool = nn.AdaptiveAvgPool2d(1)   # (B, d, 1, 1)
        self.fc_in = nn.Linear(d_in, hidden)

        # output projection (applied per sequence step)
        self.fc_out = nn.Linear(hidden, d_out)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        # x: (B, d, H, W)
        b, d, h, w = x.shape
        x = self.pool(x).view(b, d)   # (B, d)
        x = self.act(self.fc_in(x))   # (B, hidden)

        # Repeat across seq_len
        x = x.unsqueeze(1).expand(b, seq_len, -1)  # (B, seq_len, hidden)

        # Project to output dimension
        out = self.fc_out(x)   # (B, seq_len, d_out)
        return out

class UNetWithHead(nn.Module):
    def __init__(self, in_ch: int = 2, base_ch: int = 32, latent_ch: int = 16, 
                 d_out: int = 10, hidden: int = 128):
        super().__init__()
        self.feature = UNetFeatureExtractor(in_ch=in_ch, base_ch=base_ch, latent_ch=latent_ch)
        self.head = UNetHead(d_in=latent_ch, d_out=d_out, hidden=hidden)

    @torch.no_grad()
    def extract_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature(x)   # (B, latent_ch, H, W)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        latent = self.feature(x)           # (B, latent_ch, H, W)
        out = self.head(latent, seq_len)   # (B, seq_len, d_out)
        return out
    













    
################################################################
###### UNet2D with Temporal Head
################################################################
    
class UNetHeadTemporal(nn.Module):
    """
    Input:  (B, d, H, W)
    Output: (B, seq_len, d_out)
    """
    def __init__(self, d_in: int, d_out: int, hidden: int = 128, rnn_type: str = "gru", num_layers: int = 1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)   # reduce (H,W) → (1,1)
        self.fc_in = nn.Linear(d_in, hidden)

        if rnn_type.lower() == "gru":
            self.rnn = nn.GRU(hidden, hidden, num_layers=num_layers, batch_first=True)
        elif rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(hidden, hidden, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        self.fc_out = nn.Linear(hidden, d_out)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        # x: (B, d, H, W)
        b, d, h, w = x.shape
        x = self.pool(x).view(b, d)        # (B, d)
        x = torch.tanh(self.fc_in(x))      # (B, hidden)

        # Use as initial hidden state
        x0 = x.unsqueeze(1)                # (B, 1, hidden)
        rnn_in = x0.expand(b, seq_len, -1) # repeat across time: (B, seq_len, hidden)

        out, _ = self.rnn(rnn_in)          # (B, seq_len, hidden)
        out = self.fc_out(out)             # (B, seq_len, d_out)
        return out

class UNetWithTemporalHead(nn.Module):
    def __init__(self, in_ch: int = 2, base_ch: int = 32, latent_ch: int = 16,
                 d_out: int = 10, hidden: int = 128, rnn_type: str = "gru", num_layers: int = 1):
        super().__init__()
        self.feature = UNetFeatureExtractor(in_ch=in_ch, base_ch=base_ch, latent_ch=latent_ch)
        self.head = UNetHeadTemporal(d_in=latent_ch, d_out=d_out, hidden=hidden,
                                     rnn_type=rnn_type, num_layers=num_layers)

    @torch.no_grad()
    def extract_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature(x)   # (B, latent_ch, H, W)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        latent = self.feature(x)             # (B, latent_ch, H, W)
        out = self.head(latent, seq_len)     # (B, seq_len, d_out)
        return out












    
#####################################################################
########## Conv1D Model
#####################################################################

class ConvBNAct1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DoubleConv1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct1D(in_ch, out_ch, 3, 1),
            ConvBNAct1D(out_ch, out_ch, 3, 1),
        )
    def forward(self, x):
        return self.block(x)

class UNetFeatureExtractor1D(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, latent_ch=16):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv1D(in_ch, base_ch)
        self.pool1 = nn.MaxPool1d(2, ceil_mode=True)  # ceil_mode ensures coverage

        self.enc2 = DoubleConv1D(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool1d(2, ceil_mode=True)

        # Bottleneck
        self.bottleneck = DoubleConv1D(base_ch * 2, base_ch * 4)

        # Decoder (replace ConvTranspose1d with interpolation + conv)
        self.dec2 = DoubleConv1D(base_ch * 4 + base_ch * 2, base_ch * 2)
        self.dec1 = DoubleConv1D(base_ch * 2 + base_ch, base_ch)

        # Project to latent channels
        self.proj = nn.Conv1d(base_ch, latent_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)              # (B, base_ch, N)
        e2 = self.enc2(self.pool1(e1)) # (B, 2*base_ch, N1)

        # Bottleneck
        b = self.bottleneck(self.pool2(e2)) # (B, 4*base_ch, N2)

        # Decoder with size-matching interpolation
        d2 = F.interpolate(b, size=e2.size(-1), mode="linear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # (B, 2*base_ch, N1)

        d1 = F.interpolate(d2, size=e1.size(-1), mode="linear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # (B, base_ch, N)

        latent = self.proj(d1)                      # (B, latent_ch, N)
        return latent

class UNetHead1D(nn.Module):
    """
    Input:  (B, d_in, N)
    Output: (B, seq_len, N)
    """
    def __init__(self, d_in: int, hidden: int = 128, 
                 kernel_size: int = 7, padding: int = 3, blocks: int = 1):
        super().__init__()
        self.act = nn.SiLU(inplace=True)

        # Conv processing
        self.conv1 = nn.Conv1d(d_in, hidden, kernel_size=kernel_size, padding=padding)
        layers = []
        for _ in range(blocks):
            layers += [nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=padding),
                       nn.SiLU(inplace=True)]
        self.mid = nn.Sequential(*layers)
        self.convH = nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=padding)

        # Final projection: hidden → seq_len (applied per point N)
        self.proj = nn.Linear(hidden, hidden)  # placeholder, will reshape later

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        # x: (B, d_in, N)
        out = self.act(self.conv1(x))   # (B, hidden, N)
        if len(self.mid) > 0:
            out = self.mid(out)         # (B, hidden, N)
        out = self.convH(out)           # (B, hidden, N)

        # Permute for linear layer: (B, N, hidden)
        out = out.permute(0, 2, 1)

        # Map hidden → seq_len
        proj = nn.Linear(out.size(-1), seq_len).to(out.device)
        out = proj(out)                 # (B, N, seq_len)

        # Permute back: (B, seq_len, N)
        out = out.permute(0, 2, 1)
        return out


class UNetWithHead1D(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 32, latent_ch: int = 16):
        super().__init__()
        self.feature = UNetFeatureExtractor1D(in_ch=in_ch, base_ch=base_ch, latent_ch=latent_ch)
        self.head = UNetHead1D(d_in=latent_ch)  # overshoot d_out, slice later

    @torch.no_grad()
    def extract_latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature(x)  # (B, latent_ch, N)

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        latent = self.feature(x)             # (B, latent_ch, N)
        out = self.head(latent, seq_len)     # (B, seq_len, N)
        return out









#######################################################
#### RNN Model
#######################################################


class VectorToSequenceRNN(nn.Module):
    def __init__(self, input_dim=1003, hidden_dim=512, output_dim=1003, rnn_type="gru", num_layers=1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Project input vector to initial hidden state (only for first layer)
        self.fc_init = nn.Linear(input_dim, hidden_dim)

        # Choose RNN type
        if rnn_type.lower() == "gru":
            self.rnn = nn.GRU(output_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        elif rnn_type.lower() == "lstm":
            self.rnn = nn.LSTM(output_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")

        # Output projection
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, seq_len):
        """
        x: Tensor of shape [batch_size, input_dim]
        seq_len: int, length of output sequence
        Returns: Tensor of shape [batch_size, seq_len, output_dim]
        """
        batch_size = x.size(0)

        # ---- Initial hidden state ----
        # First layer initialized from fc_init(x)
        h0_first = torch.tanh(self.fc_init(x)).unsqueeze(0)  # [1, B, H]

        # Remaining layers start at zero
        h0_rest = torch.zeros(self.num_layers - 1, batch_size, self.hidden_dim, device=x.device)

        h0 = torch.cat([h0_first, h0_rest], dim=0)  # [num_layers, B, H]

        if isinstance(self.rnn, nn.LSTM):
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=x.device)
            h0 = (h0, c0)  # (h, c) for LSTM

        # ---- Sequence generation ----
        start_token = torch.zeros(batch_size, 1, self.output_dim, device=x.device)

        outputs = []
        inp = start_token

        for _ in range(seq_len):
            out, h0 = self.rnn(inp, h0)   # out: [B, 1, H]
            vec = self.fc_out(out)        # [B, 1, output_dim]
            outputs.append(vec)
            inp = vec                     # autoregressive input

        return torch.cat(outputs, dim=1)  # [B, seq_len, output_dim]

