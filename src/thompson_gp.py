"""
Thompson sampling for Bayesian optimization on Gaussian Process.

Thompson sampling: sample from the GP posterior, then act greedily on the sample.
Key insight: optimism-in-face-of-uncertainty (OIFU) for exploration-exploitation balance.
"""
import numpy as np
from scipy.linalg import cholesky


def sample_from_gp_posterior(gp, X_star, n_samples=1, jitter=1e-6):
    """
    Draw joint sample(s) from GP posterior over X_star.
    
    Procedure:
    1. Compute posterior mean μ and covariance Σ at X_star
    2. Add jitter to diagonal for numerical stability
    3. Cholesky factorize Σ + jitter*I
    4. Draw z ~ N(0,I)^{n_points × n_samples}
    5. Return f = μ + L @ z
    
    Args:
        gp: fitted GaussianProcess
        X_star: test points (n_points, n_features)
        n_samples: number of joint samples
        jitter: regularization on cov diagonal
        
    Returns:
        If n_samples == 1: shape (n_points,)
        If n_samples > 1: shape (n_points, n_samples)
    """
    mu, var = gp.predict(X_star)  # var is shape (n_points,)
    n_points = len(X_star)
    
    # Construct covariance matrix from posterior marginal variances
    # This is a *diagonal* approximation to the true posterior covariance
    # For a true joint sample, we need the full covariance matrix
    # Recompute from scratch using the GP's kernel
    K_star = gp.kernel(X_star, X_star)
    K_Xs = gp.kernel(X_star, gp.X)
    
    # Posterior covariance: Σ = K_** - K_*X (K_XX + noise*I)^{-1} K_X*
    from scipy.linalg import cho_solve
    v = cho_solve(gp.L, K_Xs.T)  # gp.L is Cholesky of K_XX + noise*I
    Sigma = K_star - K_Xs @ v
    
    # Add jitter to diagonal for stability
    Sigma_jittered = Sigma + jitter * np.eye(n_points)
    
    # Cholesky of posterior cov
    try:
        L_post = cholesky(Sigma_jittered, lower=True)
    except np.linalg.LinAlgError:
        # If still singular, add more jitter
        Sigma_jittered = Sigma + (jitter * 10) * np.eye(n_points)
        L_post = cholesky(Sigma_jittered, lower=True)
    
    # Draw samples
    z = np.random.randn(n_points, n_samples)
    samples = mu[:, None] + L_post @ z  # (n_points, n_samples)
    
    if n_samples == 1:
        return samples.flatten()
    else:
        return samples


def thompson_sampling_loop(gp, X_pool, y_pool, X_seen, y_seen, n_rounds=20, seed=42):
    """
    Run Thompson sampling bandit loop.
    
    At each round:
    1. Sample from GP posterior f ~ N(μ, Σ) over all pool arms
    2. Pick arm* = argmax_i f[i]
    3. Observe reward y[arm*] from the pool (true reward)
    4. Add (X[arm*], y[arm*]) to training set
    5. Refit GP
    6. Track cumulative regret
    
    Args:
        gp: fitted GaussianProcess (initialized with X_seen, y_seen)
        X_pool: all arm features (n_arms, n_features)
        y_pool: true arm rewards (n_arms,)
        X_seen: already observed features (n_seed,)
        y_seen: already observed rewards (n_seed,)
        n_rounds: number of Thompson rounds
        seed: random seed
        
    Returns:
        chosen_idx: list of arm indices chosen each round
        reward_history: list of rewards observed each round
        regret_history: cumulative regret each round
    """
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    
    # Best possible reward (oracle)
    y_max = np.max(y_pool)
    
    chosen_idx = []
    reward_history = []
    regret_history = []
    cum_regret = 0.0
    
    # Copy training set to avoid mutation
    X_train = X_seen.copy()
    y_train = y_seen.copy()
    
    for t in range(n_rounds):
        # Sample from posterior
        f_sample = sample_from_gp_posterior(gp, X_pool, n_samples=1, jitter=1e-6)
        
        # Greedy selection
        arm_star = np.argmax(f_sample)
        chosen_idx.append(arm_star)
        
        # Observe reward
        reward = y_pool[arm_star]
        reward_history.append(reward)
        
        # Compute regret
        instant_regret = y_max - reward
        cum_regret += instant_regret
        regret_history.append(cum_regret)
        
        # Add to training set and refit
        X_train = np.vstack([X_train, X_pool[[arm_star]]])
        y_train = np.append(y_train, reward)
        
        gp.fit(X_train, y_train)
    
    return chosen_idx, reward_history, regret_history


def random_baseline(X_pool, y_pool, n_rounds=20, seed=42):
    """
    Random arm selection baseline for comparison.
    
    Args:
        X_pool: all arm features (n_arms, n_features)
        y_pool: true arm rewards (n_arms,)
        n_rounds: number of rounds
        seed: random seed
        
    Returns:
        chosen_idx: list of arm indices chosen each round
        reward_history: list of rewards observed each round
        regret_history: cumulative regret each round
    """
    rng = np.random.default_rng(seed)
    
    # Best possible reward
    y_max = np.max(y_pool)
    
    chosen_idx = []
    reward_history = []
    regret_history = []
    cum_regret = 0.0
    
    for t in range(n_rounds):
        # Random arm
        arm = rng.integers(0, len(X_pool))
        chosen_idx.append(arm)
        
        # Observe reward
        reward = y_pool[arm]
        reward_history.append(reward)
        
        # Compute regret
        instant_regret = y_max - reward
        cum_regret += instant_regret
        regret_history.append(cum_regret)
    
    return chosen_idx, reward_history, regret_history
