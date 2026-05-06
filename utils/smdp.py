from typing import Optional, Tuple

def smdp_return_range(r_min, r_max, L, H, gamma1, gamma2):
    """
    Compute theoretical min/max return for SMDP with two discounts (gamma1, gamma2).

    Args:
        r_min: minimum reward per primitive step
        r_max: maximum reward per primitive step
        L: option length
        H: episode length (fixed)
        gamma1: intra-option discount
        gamma2: inter-option discount

    Returns:
        (G_min, G_max)
    """
    m = H // L  # number of full options
    r = H % L  # remainder length

    # geometric sums
    S_L = (1 - gamma1**L) / (1 - gamma1) if gamma1 != 1 else L
    S_r = (1 - gamma1**r) / (1 - gamma1) if (r > 0 and gamma1 != 1) else r

    # inter-option weight
    weight = (1 - gamma2 ** (m * L)) / (1 - gamma2**L)

    total_weight = S_L * weight
    if r > 0:
        total_weight += (gamma2 ** (m * L)) * S_r

    G_min = r_min * total_weight
    G_max = r_max * total_weight

    return G_min, G_max


def geometric_sum(gamma: float, n: int) -> float:
    """Compute geometric sum: sum_{i=0}^{n-1} gamma^i.
    
    Args:
        gamma: Discount factor
        n: Number of terms to sum
        
    Returns:
        Sum of geometric series. Returns n if gamma == 1.
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError('n must be non-negative')
    if n == 0:
        return 0.0
    if gamma == 1.0:
        return float(n)
    return (1.0 - gamma**n) / (1.0 - gamma)


def rtg_range_smdp(
    r_min: float = -15.0,
    r_max: float = 0.0, 
    gamma_in: float = 0.99,
    gamma_out: float = 0.99,
    L: int = 16
) -> Tuple[Optional[float], Optional[float]]:
    """Compute min/max range of infinite-horizon return-to-go for SMDP with fixed option length L.
    
    The return-to-go is defined as:
      G = sum_{k=0}^∞ (gamma_out)^k [ sum_{i=0}^{L-1} (gamma_in)^i * r_{k,i} ]
    where r_{k,i} ∈ [r_min, r_max]
    
    Args:
        r_min: Minimum reward per primitive step
        r_max: Maximum reward per primitive step  
        gamma_in: Intra-option discount factor
        gamma_out: Inter-option discount factor
        L: Fixed option length
        
    Returns:
        Tuple (G_min, G_max) where:
        - Both values are finite if gamma_out < 1
        - Values may be None (infinite) if gamma_out == 1 and rewards are non-zero
        - Values are (0.0, 0.0) if gamma_out == 1 and r_min = r_max = 0
        
    Raises:
        ValueError: If L < 0, r_min > r_max, or gamma_out > 1
    """
    if L < 0:
        raise ValueError('L must be non-negative')
    if r_min > r_max:
        raise ValueError('r_min must be <= r_max')
    if gamma_out > 1.0:
        raise ValueError('gamma_out must be <= 1 for convergent infinite-horizon return')

    # Compute inner sum over option steps
    inner_sum = geometric_sum(gamma_in, L)

    # Handle convergent case
    if gamma_out < 1.0:
        outer_factor = 1.0 / (1.0 - gamma_out)
        G_min = r_min * inner_sum * outer_factor
        G_max = r_max * inner_sum * outer_factor
        return G_min, G_max

    # Handle gamma_out == 1 case
    G_max = None if r_max > 0 else (0.0 if r_max == 0.0 else None)
    G_min = None if r_min < 0 else (0.0 if r_min == 0.0 else None)
    return G_min, G_max

if __name__ == '__main__':
    r_min, r_max = -4, 0
    L = 4
    H = 1000
    gamma_in = 0.9
    gamma_out = 0.995
    single_min = r_min * (1 - gamma_in**L) / (1 - gamma_in)
    single_max = r_max * (1 - gamma_in**L) / (1 - gamma_in)
    print(f'single_min: {single_min}, single_max: {single_max}')
    r_range = smdp_return_range(r_min, r_max, L, H, gamma_in, gamma_out)
    print(f'r_range: {r_range}')
    G_min, G_max = rtg_range_smdp(r_min, r_max, gamma_in, gamma_out, L)
    print(f'G_min: {G_min}, G_max: {G_max}')
