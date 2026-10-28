import torch
import numpy as np

def generate_synthetic_mhd_data(num_samples=100, grid_size=64, time_steps=20):
    """
    Generates synthetic Reduced Magnetohydrodynamic (RMHD) field evolution 
    representing magnetic island growth and tearing mode precursors.
    """
    print(f"Generating synthetic RMHD dataset: {num_samples} samples, {grid_size}x{grid_size} grid...")
    
    x = np.linspace(0, 2 * np.pi, grid_size)
    y = np.linspace(0, 2 * np.pi, grid_size)
    X, Y = np.meshgrid(x, y)
    
    dataset_inputs = []
    dataset_targets = []
    
    for _ in range(num_samples):
        k_x = np.random.choice([1, 2])
        k_y = np.random.choice([1, 2])
        growth_rate = np.random.uniform(0.05, 0.15)
        
        sample_sequence_in = []
        
        for t in range(time_steps):
            t_val = t * 0.1
            psi = np.sin(k_x * X) * np.cos(k_y * Y) * np.exp(growth_rate * t_val)
            omega = -0.5 * np.cos(k_x * X) * np.sin(k_y * Y) * np.exp(growth_rate * t_val)
            
            psi += np.random.normal(0, 0.02, size=X.shape)
            omega += np.random.normal(0, 0.02, size=X.shape)
            
            field_state = np.stack([psi, omega], axis=-1)
            sample_sequence_in.append(field_state)
            
        inputs = np.array(sample_sequence_in[:-1])
        targets = np.array(sample_sequence_in[1:])
        
        dataset_inputs.append(inputs)
        dataset_targets.append(targets)
        
    return torch.tensor(np.array(dataset_inputs), dtype=torch.float32), \
           torch.tensor(np.array(dataset_targets), dtype=torch.float32)

if __name__ == "__main__":
    inputs, targets = generate_synthetic_mhd_data(num_samples=10)
    print("Data generation successful!")
    print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")