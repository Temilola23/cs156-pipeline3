# TVAE ELBO Derivation

## Evidence Lower Bound (ELBO)

We seek to maximize the log-likelihood $\log p(x)$ of the data. Using the variational inference framework with an encoder distribution $q(z|x)$ and prior $p(z)$:

$$\log p(x) = \log \int p(x|z) p(z) dz$$

By introducing $q(z|x)$ and applying Jensen's inequality:

$$\log p(x) = \log \mathbb{E}_{z \sim q(z|x)} \left[ \frac{p(x, z)}{q(z|x)} \right]$$

$$\geq \mathbb{E}_{z \sim q(z|x)} \left[ \log \frac{p(x, z)}{q(z|x)} \right]$$

$$= \mathbb{E}_{z \sim q(z|x)} \left[ \log p(x|z) \right] - \mathbb{E}_{z \sim q(z|x)} \left[ \log \frac{q(z|x)}{p(z)} \right]$$

$$= \mathbb{E}_{z \sim q} \left[ \log p(x|z) \right] - KL(q(z|x) \parallel p(z))$$

The right-hand side is the **Evidence Lower Bound (ELBO)**.

## ELBO in Practice

### 1. Reconstruction Loss

Assuming a Gaussian decoder with variance 1:

$$\mathbb{E}_{z \sim q} \left[ \log p(x|z) \right] \approx -\frac{1}{2N} \sum_{i=1}^{N} \|x_i - \mu_{\text{dec}}(z_i)\|^2$$

where $\mu_{\text{dec}}(z)$ is the decoder output (reconstruction). This is the **Mean Squared Error (MSE)** loss.

### 2. KL Divergence

With $q(z|x) = \mathcal{N}(z; \mu_{\text{enc}}(x), \sigma_{\text{enc}}^2(x) I)$ and prior $p(z) = \mathcal{N}(0, I)$:

$$KL(q(z|x) \parallel p(z)) = \frac{1}{2N} \sum_{i=1}^{N} \sum_{j=1}^{z_{\text{dim}}} \left[ 1 + \log\sigma_{ij}^2 - \mu_{ij}^2 - \sigma_{ij}^2 \right]$$

We parameterize via $\log_{\text{var}} = \log(\sigma^2)$, so:

$$KL = \frac{1}{2N} \sum_{i,j} \left[ 1 + \log\text{var}_{ij} - \mu_{ij}^2 - \exp(\log\text{var}_{ij}) \right]$$

### 3. Total Loss

$$\text{ELBO} = \text{MSE}(x, \hat{x}) + \beta \cdot KL(q \parallel p)$$

where $\beta$ is a weighting factor (often set to 1.0 or annealed during training).

## Reparameterization Trick

To enable backpropagation through the sampling operation:

$$z = \mu(x) + \sigma(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

where $\odot$ is element-wise multiplication and $\sigma(x) = \exp(0.5 \log\text{var}(x))$.

## Implementation Notes

- **Encoder**: Maps $x \in \mathbb{R}^d \to (\mu, \log\text{var}) \in \mathbb{R}^{z_{\text{dim}}} \times \mathbb{R}^{z_{\text{dim}}}$
- **Decoder**: Maps $z \in \mathbb{R}^{z_{\text{dim}}} \to \hat{x} \in \mathbb{R}^d$ (reconstruction)
- **Loss computation**: Minimizing negative ELBO
- **KL annealing** (optional): Gradually increase $\beta$ from 0 to 1 during training to avoid posterior collapse

## References

Kingma, D. P., & Welling, M. (2013). "Auto-encoding variational Bayes." arXiv preprint arXiv:1312.6114.
