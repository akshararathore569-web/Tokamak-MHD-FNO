import torch
import torch.nn as nn

class RMHDEnergyLoss(nn.Module):
    def __init__(self, lambda_physics=0.01):
        super(RMHDEnergyLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_physics = lambda_physics # Weight for physics regularization

    def forward(self, pred, target, inputs):
        """
        Computes a combined loss: Data MSE + RMHD Physics Residual Constraint.
        
        Args:
            pred: Predicted state tensor (batch, H, W, 2) -> [psi, omega]
            target: Ground truth state tensor (batch, H, W, 2)
            inputs: Previous state tensor (batch, H, W, 2)
        """
        # 1. Standard Data-driven Loss (MSE between prediction and ground truth)
        data_loss = self.mse(pred, target)

        # 2. Physics-Informed Residual Constraint (Reduced MHD energy/flux smoothness)
        # Approximate spatial gradients using finite differences
        # psi is channel 0, omega is channel 1
        pred_psi = pred[..., 0]
        
        # Compute spatial gradients of magnetic flux (psi) to penalize unphysical sudden jumps
        dx_psi = torch.abs(pred_psi[:, :, 1:] - pred_psi[:, :, :-1])
        dy_psi = torch.abs(pred_psi[:, 1:, :] - pred_psi[:, :-1, :])
        
        # Physics residual: enforcing smooth magnetic island evolution without artificial spikes
        physics_residual = torch.mean(dx_psi) + torch.mean(dy_psi)

        # Total combined loss
        total_loss = data_loss + (self.lambda_physics * physics_residual)
        return total_loss, data_loss, physics_residual

if __name__ == "__main__":
    criterion = RMHDEnergyLoss()
    dummy_pred = torch.randn(2, 64, 64, 2, requires_grad=True)
    dummy_target = torch.randn(2, 64, 64, 2)
    dummy_input = torch.randn(2, 64, 64, 2)
    
    loss, d_loss, p_loss = criterion(dummy_pred, dummy_target, dummy_input)
    print("Physics-Informed Loss function initialized successfully!")
    print(f"Total Loss: {loss.item():.4f} (Data Loss: {d_loss.item():.4f}, Physics Residual: {p_loss.item():.4f})")