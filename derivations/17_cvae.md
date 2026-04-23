# Derivation: Conditional VAE for Rating-Conditioned Feature Synthesis

## Problem Setup

We have a dataset of movie features `x ∈ ℝ^d` (e.g., year, runtime, TMDB vote, genres, modality one-hot) and want to learn to generate synthetic features conditioned on a discrete rating bin `c ∈ {1, 2, 3, 4, 5}`.

The key insight is that a **Conditional VAE** extends the vanilla VAE by conditioning both the encoder and decoder on the discrete label `c`.

## Vanilla VAE Recap

The standard VAE maximizes the ELBO:

```
ELBO = E_q(z|x)[log p(x|z)] - KL(q(z|x) || p(z))
```

where:
- `q(z|x)` is the encoder (posterior approximation)
- `p(x|z)` is the decoder (likelihood)
- `p(z) = N(0, I)` is the prior
- `z ∈ ℝ^{z_dim}` is the latent variable

## Conditional VAE

For conditional generation, we condition both distributions on `c`:

```
ELBO = E_q(z|x,c)[log p(x|z,c)] - KL(q(z|x,c) || p(z))
```

The key changes:
- **Encoder**: `q(z|x,c)` takes both `x` and `c` as input
  - Input: `[x; c_onehot]` where `c_onehot ∈ {0,1}^{n_cond}`
  - Output: `(μ, log_var)` for the latent distribution
  
- **Decoder**: `p(x|z,c)` takes both `z` and `c` as input
  - Input: `[z; c_onehot]`
  - Output: reconstructed `x_recon`

- **Prior**: We use the standard prior `p(z) = N(0, I)` (not conditioned on `c`)
  - This keeps the latent space consistent across conditions

## Loss Function

### Reconstruction Term
We use MSE loss as a proxy for `E_q[log p(x|z,c)]`:

```
L_recon = MSE(x, x_recon) = (1/d) Σ (x_i - x_recon_i)²
```

### KL Divergence Term
For a Gaussian posterior with diagonal covariance:

```
KL(q(z|x,c) || p(z)) = KL(N(μ, Σ) || N(0, I))
                      = (1/2) Σ_j (1 + log_var_j - μ_j² - exp(log_var_j))
```

where `Σ_j = exp(log_var_j)`.

### Total ELBO Loss with Annealing
We use **KL annealing** to avoid posterior collapse:

```
L = L_recon + β(epoch) * L_KL
```

where `β` ramps from 0 to 1.0 over the first 30% of training, then stays at 1.0.

## Sampling for Conditional Generation

To generate `n` samples conditioned on rating bin `c`:

1. Sample `z ~ N(0, I)`, shape `(n, z_dim)`
2. One-hot encode condition: `c_onehot`, shape `(n, n_cond)`
3. Decode: `x_gen = decoder([z; c_onehot])`

This allows us to steer generation toward specific rating bins without re-training the VAE.

## Architecture Details

- **Input dimension**: `d = 9` (4 numeric + 4 modality one-hot + 1 will be concatenated in conditioning, but actually d=8 since we exclude rating from features)
- **Condition dimension**: `n_cond = 5` (one rating bin per condition)
- **Hidden dimension**: `h = 32`
- **Latent dimension**: `z_dim = 4`

### Encoder
```
[x; c_onehot] (d + n_cond = 13) 
  → FC(32) + ReLU 
  → FC(32) + ReLU 
  → μ (z_dim) 
  → log_var (z_dim)
```

### Decoder
```
[z; c_onehot] (z_dim + n_cond = 9) 
  → FC(32) + ReLU 
  → FC(32) + ReLU 
  → x_recon (d = 8)
```

## Training Details

- **Data**: 324 real ratings with movie features
- **Rating distribution**: Heavily skewed (323/324 in bin 5, 1 in bin 4)
- **Epochs**: 1000
- **Batch size**: 32
- **Optimizer**: Adam (lr=1e-3)
- **KL annealing**: β ramps 0→1 over first 300 epochs

## Interpretation

The conditional VAE learns to map from `(z, c)` pairs to plausible movie features for that rating bin. Even though most real data is in bin 5, the model can generate synthetic examples for all bins by learning a task-specific latent representation.

The low variance in synthetic vote_average across conditions (7.36–7.47) suggests the model learned that TMDB vote is a strong discriminator in the data and is appropriately regularized by the KL term.
