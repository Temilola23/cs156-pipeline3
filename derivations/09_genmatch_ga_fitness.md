# GenMatch GA fitness derivation

**Problem.** Observational data $X \in \mathbb{R}^{n \times d}$, binary treatment $T \in \{0,1\}^n$. Goal: match each treated unit to its nearest control unit so that — marginally across the $d$ covariates — the matched treated/control distributions are statistically indistinguishable.

**Distance.** Weighted Mahalanobis with diagonal scaling:
$d_w(x, x') = \sqrt{(x - x')^T W^{1/2} \Sigma^{-1} W^{1/2} (x - x')}$,
where $\Sigma$ is the pooled covariance and $W = \mathrm{diag}(w)$ with $w \ge 0$.

**Fitness.** For weights $w$, let $M(w)$ be the matched-control assignment obtained by nearest-neighbour search under $d_w$. Define
$F(w) = \min_{j=1..d} \mathrm{KS\text{-}pvalue}\big(X_{T=1,j},\ X_{M(w),j}\big)$.

This is Sekhon & Diamond's "minimum p-value" criterion: the GA maximises the worst-covariate p-value, which forces balance across *all* covariates simultaneously rather than averaging.

**Genetic algorithm.** Real-valued chromosomes $w \in \mathbb{R}^d_+$:
- Tournament selection (k = 3)
- Uniform per-gene crossover: child = $\alpha a + (1-\alpha) b,\ \alpha \sim U(0,1)^d$
- Gaussian mutation: each gene mutates with probability $p_m$ by $\mathcal N(0, \sigma_m)$, clipped at 0
- Elitism: top-$e$ individuals pass through unchanged → guarantees $F_t^* \ge F_{t-1}^*$

**Convergence intuition.** Each generation the best-so-far fitness is monotone non-decreasing in expectation under elitism; in practice after 20–50 generations the GA settles near a weighting that balances all covariates simultaneously. The worst-covariate objective makes the matching robust to covariate miscalibration.
