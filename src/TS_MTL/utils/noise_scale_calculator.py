from opacus.accountants.utils import get_noise_multiplier
# This function calculates the optimal noise scale for differential privacy
# based on the dataset size, batch size, local epochs, rounds, and target privacy budget
def find_optimal_noise_scales(dataset_size, batch_size, local_epochs, rounds, target_epsilon, target_delta):
    """
    Find optimal noise multiplier values for gradient privacy.
    
    Args:
        dataset_size: Total number of samples across all clients
        batch_size: Batch size used for training
        local_epochs: Number of local epochs per communication round
        rounds: Number of communication rounds
        target_epsilon: Target privacy budget (epsilon)
        target_delta: Target privacy budget (delta)
        
    Returns:
        Dictionary containing recommended noise scale values
    """
    # Calculate sampling rate
    sample_rate = batch_size / dataset_size
    
    # Calculate total steps
    steps = local_epochs * rounds
    
    # Find the optimal noise multiplier
    optimal_sigma = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=sample_rate,
        steps=steps,
        accountant='rdp'  # Use RDP accountant as in your implementation
    )
    
    print(f"For ε={target_epsilon}, δ={target_delta}, steps={steps}, sample_rate={sample_rate}:")
    print(f"Optimal noise multiplier: {optimal_sigma:.4f}")
    
    # Generate a range of noise values to test
    sigma_s_values = [
        optimal_sigma * 0.5,  # Lower noise (less privacy, better performance)
        optimal_sigma,        # Optimal noise for target epsilon
        optimal_sigma * 2.0   # Higher noise (more privacy, worse performance)
    ]
    
    # Encoder noise is typically lower than gradient noise
    sigma_e_values = [
        optimal_sigma * 0.02,  # Very low encoder noise
        optimal_sigma * 0.05,  # Low encoder noise
        optimal_sigma * 0.1    # Moderate encoder noise
    ]
    
    return {
        "optimal_sigma": optimal_sigma,
        "sigma_s_values": sigma_s_values,
        "sigma_e_values": sigma_e_values
    }