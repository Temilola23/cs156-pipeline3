# HMM forward-backward in log-space

## Model

Hidden state $z_t \in \{1,...,K\}$, discrete observation $x_t \in \{1,...,B\}$.

**Parameters:**
- Transition matrix: $A_{ij} = P(z_t = j | z_{t-1} = i)$ (K × K)
- Emission matrix: $E_{ib} = P(x_t = b | z_t = i)$ (K × B)
- Initial distribution: $\pi_i = P(z_1 = i)$ (K)

## Forward Algorithm

**Definition:** $\alpha_t(i) = P(x_{1:t}, z_t = i)$ (joint probability of observations and state)

**Log-space recurrence:**

$$\log \alpha_1(i) = \log \pi_i + \log E_{i, x_1}$$

$$\log \alpha_t(i) = \log E_{i, x_t} + \mathrm{logsumexp}_j(\log \alpha_{t-1}(j) + \log A_{ji})$$

where $\mathrm{logsumexp}_j(a_j) = \max_j(a_j) + \log \sum_j \exp(a_j - \max_k(a_k))$ for numerical stability.

## Backward Algorithm

**Definition:** $\beta_t(i) = P(x_{t+1:T} | z_t = i)$ (likelihood of future observations given state)

**Log-space recurrence:**

$$\log \beta_T(i) = 0$$

$$\log \beta_t(i) = \mathrm{logsumexp}_j(\log A_{ij} + \log E_{j, x_{t+1}} + \log \beta_{t+1}(j))$$

## Likelihood

**Total observation likelihood:**

$$\log P(x_{1:T}) = \mathrm{logsumexp}_i(\log \alpha_T(i))$$

This can also be computed via $\beta_1$ in initial frame (numerical validation).

## Posterior State Occupancy

**State posterior (filtering distribution):**

$$\gamma_t(i) = P(z_t = i | x_{1:T}) = \frac{\alpha_t(i) \beta_t(i)}{P(x_{1:T})}$$

In log-space:
$$\log \gamma_t(i) = \log \alpha_t(i) + \log \beta_t(i) - \log P(x_{1:T})$$

## State Transition Posterior

**Joint state posterior:**

$$\xi_t(i,j) = P(z_t = i, z_{t+1} = j | x_{1:T}) = \frac{\alpha_t(i) A_{ij} E_{j,x_{t+1}} \beta_{t+1}(j)}{P(x_{1:T})}$$

In log-space:
$$\log \xi_t(i,j) = \log \alpha_t(i) + \log A_{ij} + \log E_{j, x_{t+1}} + \log \beta_{t+1}(j) - \log P(x_{1:T})$$

## Baum-Welch (EM) M-Step

Update parameters using expected counts:

**Initial state distribution:**
$$\pi_i \leftarrow \gamma_1(i)$$

**Transition matrix:**
$$A_{ij} \leftarrow \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$$

**Emission matrix:**
$$E_{ib} \leftarrow \frac{\sum_{t: x_t = b} \gamma_t(i)}{\sum_{t=1}^{T} \gamma_t(i)}$$

After each update, normalize to ensure distributions sum to 1.

## Viterbi Algorithm (Most Likely Path)

**Definition:** $\delta_t(i) = \max_{z_{1:t-1}} P(z_{1:t} = (..., i), x_{1:t})$ (max log-probability)

**Recurrence:**
$$\delta_1(i) = \log \pi_i + \log E_{i, x_1}$$

$$\delta_t(i) = \max_j(\delta_{t-1}(j) + \log A_{ji}) + \log E_{i, x_t}$$

**Backpointer:** $\psi_t(i) = \arg\max_j(\delta_{t-1}(j) + \log A_{ji})$

**Backtrack:** $z_T^* = \arg\max_i \delta_T(i)$, then $z_t^* = \psi_{t+1}(z_{t+1}^*)$

## Numerical Stability Notes

- All computations performed in **log-space** to avoid underflow on long sequences
- Use $\mathrm{logsumexp}$ with max-subtraction trick for log-domain addition
- Clip small probabilities to $10^{-300}$ before taking logarithm
- All distributions (A, E, π) stored as normalized probabilities; convert to log only when needed
