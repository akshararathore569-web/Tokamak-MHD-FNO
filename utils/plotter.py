import torch
import matplotlib.pyplot as plt
import os

def plot_mhd_results(model, test_input, test_target, save_path="results_comparison.png"):
    """
    Generates a 3-panel side-by-side comparison figure:
    1. Ground Truth Magnetic Flux
    2. FNO Prediction
    3. Absolute Physics Residual / Error Map
    """
    model.eval()
    with torch.no_grad():
        # Forward pass on a single sample frame
        pred = model(test_input)
        
    # Extract channel 0 (e.g., magnetic flux psi) for visualization
    # Shape handling: tensor batch/spatial extraction
    gt_np = test_target[0, :, :, 0].cpu().numpy()
    pred_np = pred[0, :, :, 0].cpu().numpy()
    error_np = torch.abs(test_target[0, :, :, 0] - pred[0, :, :, 0]).cpu().numpy()
    
    # Plotting configuration
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im0 = axes[0].imshow(gt_np, cmap='plasma', origin='lower')
    axes[0].set_title("Ground Truth (RMHD Flux)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    
    im1 = axes[1].imshow(pred_np, cmap='plasma', origin='lower')
    axes[1].set_title("FNO Prediction")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    im2 = axes[2].imshow(error_np, cmap='inferno', origin='lower')
    axes[2].set_title("Absolute Error Map")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Visual diagnostic plot successfully saved to {save_path}")