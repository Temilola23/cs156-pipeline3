# Task 1.14: Thompson Sampling on Gaussian Process

## Motivation

**Thompson Sampling** is a Bayesian approach to the exploration-exploitation trade-off in multi-armed bandits. The key insight is **optimism in the face of uncertainty**: at each round, sample from your *posterior* belief about arm rewards, then act greedily on the sample.

## Setting

- **Arms**: 39 movies (pool) with unknown rewards
- **Reward**: Temilola's true mean rating for each movie
- **Observations**: Started with 20 "seed" movies already rated, added new arms via bandit pulls
- **Features**: Per-movie features (year, runtime, vote_average, n_genres) — z-scored
- **Posterior**: Gaussian Process trained on observations

## Thompson Sampling Algorithm

At round $t$:

1. **Sample from posterior**: Draw a joint sample $f \sim N(\mu_t, \Sigma_t)$ where:
   - $\mu_t(x) = \mathbb{E}[\text{reward}(x) | \text{data}_t]$ (GP posterior mean)
   - $\Sigma_t(x, x') = \text{Cov}[\text{reward}(x), \text{reward}(x') | \text{data}_t]$ (GP posterior covariance)
   - The sample is **joint** over all 39 arms, not independent per-arm
   - Use Cholesky decomposition: $L L^\top = \Sigma_t + \epsilon I$, then $f = \mu_t + L z$ where $z \sim N(0, I)$

2. **Greedy selection**: Pick $a^* = \arg\max_i f[i]$

3. **Observe reward**: Get true reward $y = y_{a^*}$ (look up from ground truth)

4. **Update training set**: Add $(x_{a^*}, y_{a^*})$ to observations

5. **Refit GP**: Recompute $\mu_t, \Sigma_t$ on enlarged dataset

6. **Compute regret**: $r_t = \max_i y_i - y_{a^*}$ (instant regret), accumulate to cumulative regret

## Why This Works

The posterior distribution captures **uncertainty** about each arm:
- Arms with few observations → high posterior variance → wider confidence intervals
- Arms consistent with the data → low posterior variance → confident predictions

By sampling from the posterior and acting greedily, Thompson sampling naturally **explores** high-variance (uncertain) arms and **exploits** low-variance (confident) high-mean arms.

Formally, this balances exploration-exploitation without explicit $\epsilon$ schedules:
$$\Pr[\text{sample arm} \mid \text{data}] \approx \Pr[\text{arm is optimal} \mid \text{data}]$$

## Regret Bound (Sketch)

Under standard assumptions (sub-Gaussian rewards, bounded feature norms), Thompson sampling achieves:
$$\mathbb{E}[R_T] = O(\sqrt{d T \log T})$$

where $d$ is feature dimension and $T$ is number of rounds. This is near-optimal (matches lower bounds up to log factors).

## Implementation Notes

- **Cholesky stability**: Add jitter ($10^{-6}$ to diagonal) to avoid numerical issues
- **GP refitting**: Refit after each observation (feasible for 39 arms × 20 rounds = 780 GPs)
- **Kernel choice**: RBF with length=1.0, variance=1.0; noise=0.1 on training covariance
- **Joint vs independent**: Critical that we sample from the **joint** posterior, not per-arm. This encodes correlation structure learned by the GP

## Results on 82 Movies

**Experiment**: 
- 59 movies with complete features (year, runtime, vote_average, n_genres)
- 20 random seed observations
- 39 bandit arms
- 20 rounds, averaged over 10 random seeds

**Performance**:
- Thompson cumulative regret: **3.99 ± 1.10**
- Random baseline regret: **17.42 ± 1.68**
- **Advantage**: 13.42 regret reduction (77% improvement)

**Top chosen arms** (across all seeds):
1. Arm 16: 8.675 rating (pulled 169/200 times = dominant arm)
2. Arm 0: 8.300 rating (pulled 12 times)
3. Arm 14: 7.275 rating (pulled 6 times)

Arm 16 is the true max in the pool → Thompson quickly identified and exploited it. The steep curve vs random shows strong early learning.

## Extensions

1. **Contextual Thompson**: If new movies arrive without ratings, use features to initialize prior
2. **Batch parallel**: Sample multiple Thompson arms, run in parallel, aggregate posterior
3. **Hyperparameter tuning**: Optimize GP kernel hyperparameters on held-out split
4. **Utility functions**: Swap reward for a non-linear utility (e.g., log-rating)
