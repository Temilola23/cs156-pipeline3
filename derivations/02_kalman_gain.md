# Kalman gain derivation

Linear-Gaussian state-space:
$x_t = F x_{t-1} + w_t,\ w_t \sim \mathcal N(0, Q)$
$y_t = H x_t + v_t,\ v_t \sim \mathcal N(0, R)$

Posterior after observing $y_t$ is Gaussian. Matching moments of
$p(x_t | y_{1:t}) \propto p(y_t | x_t) p(x_t | y_{1:t-1})$
gives the update equations. The gain $K = P H^T (H P H^T + R)^{-1}$ minimises posterior MSE; see Murphy 2012 §18.3 for full derivation.

$x_t \leftarrow x_t + K(y_t - H x_t)$
$P_t \leftarrow (I - K H) P_t$
