# UKF sigma-point weights (Merwe scaled)

For an $n$-dimensional state, $2n+1$ sigma points capture mean+covariance under nonlinear transforms to 3rd-order Taylor accuracy. Merwe's scaled set:

$\lambda = \alpha^2(n+\kappa) - n$, defaults $\alpha=10^{-3}, \beta=2, \kappa=0$.
Sigma points: $\chi_0 = x$, $\chi_i = x \pm (\sqrt{(n+\lambda)P})_i$.
Weights:
$W^{(m)}_0 = \lambda/(n+\lambda)$
$W^{(c)}_0 = \lambda/(n+\lambda) + (1 - \alpha^2 + \beta)$
$W^{(m)}_i = W^{(c)}_i = 1/(2(n+\lambda))$ for $i \ge 1$.
$\beta=2$ is optimal for Gaussian priors (incorporates prior knowledge of kurtosis).
