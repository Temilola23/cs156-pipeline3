# MC Dropout as Variational Inference in Deep GPs

## Connection (Gal & Ghahramani 2016)

A standard neural network with **dropout kept active at test time** approximates Bayesian inference in a deep Gaussian Process.

### Key Insight

When we apply dropout with mask $\mathbf{M}_t$ (where $M_{ij} \sim \text{Bernoulli}(1-p)$) to weight matrices $\mathbf{W}$, we are sampling from a distribution over weight configurations:

$$\mathbf{W}_t = \text{diag}(\mathbf{M}_t) \mathbf{W}$$

Running $T$ forward passes with different dropout masks gives us $T$ stochastic samples from the posterior predictive distribution.

### Predictive Distribution

The posterior predictive distribution over outputs given input $\mathbf{x}$ is approximated by:

$$p(y | \mathbf{x}) \approx \frac{1}{T} \sum_{t=1}^{T} f(\mathbf{x}; \mathbf{W}_t)$$

where $\mathbf{W}_t$ is the weight matrix with dropout mask applied.

### Mean and Variance Estimates

From $T$ forward passes $\{y_t\}_{t=1}^{T}$:

**Predictive mean:**
$$\mu(\mathbf{x}) = \frac{1}{T} \sum_{t=1}^{T} f(\mathbf{x}; \mathbf{W}_t)$$

**Predictive variance (sample-based, ignoring aleatoric term):**
$$\sigma^2(\mathbf{x}) = \frac{1}{T} \sum_{t=1}^{T} f(\mathbf{x}; \mathbf{W}_t)^2 - \mu(\mathbf{x})^2$$

This variance estimate captures **epistemic uncertainty** (model uncertainty) due to the distribution over network weights.

## Why It Works

1. **Variational Approximation:** Dropout induces a variational distribution $q(\mathbf{W})$ over weights that differs from the prior $p(\mathbf{W})$.

2. **ELBO Lower Bound:** Training with dropout-regularized loss minimizes a variational upper bound on the classification/regression error, which is equivalent to maximizing the ELBO (Evidence Lower Bound) in a Bayesian interpretation.

3. **Posterior Sampling:** Each stochastic forward pass samples from the approximate posterior distribution $q(\mathbf{y} | \mathbf{x})$.

## Advantages

- **No retraining required:** Use an already-trained network; just keep dropout active.
- **Computational efficiency:** $T$ forward passes is cheap compared to methods like MCMC.
- **Scalability:** Works on large deep networks (unlike many exact Bayesian methods).

## Limitations

- Provides only an **approximation** to true Bayesian inference.
- The uncertainty estimates depend heavily on the dropout rate and training procedure.
- Does not distinguish between aleatoric (data) and epistemic (model) uncertainty explicitly.

## Application to Movie Ratings

In Task 1.11, we use MC Dropout to estimate predictive uncertainty in a 2-layer neural network predicting Temilola's movie ratings from features (year, runtime, TMDB vote, genres, modality).

- **Model:** Simple MLP with dropout at $p=0.2$ between layers.
- **Inference:** $T=50$ forward passes on each test sample.
- **Uncertainty:** Standard deviation across the 50 predictions indicates which movie-modality pairs the model is most unsure about.
- **Interpretation:** High uncertainty could indicate:
  - The sample is far from training data.
  - Multiple plausible predictions exist (multimodal posterior).
  - The rating may be genuinely ambiguous for that movie/modality combination.

