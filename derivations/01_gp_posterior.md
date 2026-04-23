# GP posterior via Cholesky

Given training pairs $(X, y)$, kernel $k$, noise $\sigma^2$:
$p(f_* | X, y, X_*) = \mathcal{N}(\mu_*, \Sigma_*)$ where
$\mu_* = K_{*X}(K + \sigma^2 I)^{-1} y$
$\Sigma_* = K_{**} - K_{*X}(K+\sigma^2 I)^{-1} K_{X*}$

Numerical stability: Cholesky $K + \sigma^2 I = LL^T$, solve $\alpha = L^{-T} L^{-1} y$ once.
LML: $\log p(y|X) = -\frac12 y^T\alpha - \sum \log L_{ii} - \frac n 2 \log 2\pi$.
