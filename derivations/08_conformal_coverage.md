# Split conformal prediction — marginal coverage guarantee

**Setup.** Data $(X_i, Y_i)$ iid from $P$. Split into proper-train $\mathcal D_1$ and calibration $\mathcal D_2$ (size $n$). Fit any base regressor $\hat f$ on $\mathcal D_1$. Compute residual scores $R_i = |Y_i - \hat f(X_i)|$ for $i \in \mathcal D_2$.

Let $\hat q = R_{(\lceil(1-\alpha)(n+1)\rceil)}$, the $\lceil(1-\alpha)(n+1)\rceil$-th order statistic of the calibration residuals.

**Prediction interval:** $C(X_{n+1}) = [\hat f(X_{n+1}) - \hat q,\ \hat f(X_{n+1}) + \hat q]$.

**Theorem (Vovk, Shafer, Lei).** For an exchangeable test point $(X_{n+1}, Y_{n+1})$:
$$\Pr(Y_{n+1} \in C(X_{n+1})) \ge 1 - \alpha.$$

**Proof sketch.** By exchangeability, the rank of $R_{n+1}$ among $\{R_1, \ldots, R_{n+1}\}$ is uniform on $\{1, \ldots, n+1\}$. Hence $\Pr(R_{n+1} \le \hat q) \ge \lceil(1-\alpha)(n+1)\rceil / (n+1) \ge 1-\alpha$.

Guarantee is **distribution-free** (no assumption on $\hat f$ or $P$) and **marginal** (not conditional on $X$). Upper bound $1 - \alpha + 1/(n+1)$ holds when scores have no ties.
