import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply in frequency space (height)
        self.modes2 = modes2  # Number of Fourier modes to multiply in frequency space (width)

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # Compute 2D Fourier transform
        x_ft = torch.fft.rfftn(x, dim=[-2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space via inverse 2D Fourier transform
        x = torch.fft.irfftn(out_ft, s=(x.size(-2), x.size(-1)), dim=[-2, -1])
        return x

class TokamakFNO2d(nn.Module):
    def __init__(self, modes=12, width=32, in_channels=2, out_channels=2):
        super(TokamakFNO2d, self).__init__()
        self.modes = modes
        self.width = width
        
        # Lifting layer: maps input channels (psi, omega) to higher dimensional feature space
        self.fc0 = nn.Linear(in_channels + 2, self.width) # +2 for coordinate channels (x, y)

        # Fourier layers
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes, self.modes)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)

        # Projection layer: maps back to physical output channels
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        # x shape: (batch, H, W, in_channels)
        batchsize, H, W, _ = x.shape
        
        # Add grid coordinates (mesh grids) as auxiliary input features for spatial grounding
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)

        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2) # (batch, channels, H, W)

        # FNO Layer 1
        x1 = self.conv0(x) + self.w0(x)
        x = F.gelu(x1)

        # FNO Layer 2
        x1 = self.conv1(x) + self.w1(x)
        x = F.gelu(x1)

        # FNO Layer 3
        x1 = self.conv2(x) + self.w2(x)
        x = x

        x = x.permute(0, 2, 3, 1) # (batch, H, W, channels)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y, _ = shape
        gradx = torch.linspace(0, 1, size_x, device=device)
        gradx = gradx.reshape(1, size_x, 1, 1).repeat(batchsize, 1, size_y, 1)
        grady = torch.linspace(0, 1, size_y, device=device)
        grady = grady.reshape(1, 1, size_y, 1).repeat(batchsize, size_x, 1, 1)
        return torch.cat((gradx, grady), dim=-1)

if __name__ == "__main__":
    model = TokamakFNO2d()
    dummy_input = torch.randn(2, 64, 64, 2)
    output = model(dummy_input)
    print("FNO Model forward pass successful!")
    print(f"Output shape: {output.shape}")