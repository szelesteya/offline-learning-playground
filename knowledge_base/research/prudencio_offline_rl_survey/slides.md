# Notations

# Taxonomy

- Learning model
  - Model dynamics ${p_{\psi_T}}$
  - Trajectory distribution ${p_{\pi_{B}}}$
- Learning policy distribution ${\pi_{\theta}(a|s)}$
  - Actor-Critic - by steps taken for evaluation and optimization
    - One-step methods
    - Multi-step methods
  - Imitation Learning - mimic remaining samples from ${\pi_B}$
- Loss-terms
  - Policy constraing $\mathbb{E}_{s \sim d^{\pi_{\theta}}(\cdot), a\sim\pi_{\theta}(\cdot|s)}[Q^{\pi}(s, a)]$ s.t. $D(\pi_{\theta}(\cdot|s), \hat{\pi}_{\beta}(\cdot|s)) \leq \epsilon$
    - Requirement: $\hat{\pi}_{\beta}(\cdot|s)$ estimate
    - Disadvantages: might constrain to policy in places which it is clearly suboptimal
    - Variations
      - By the estimate ${\pi_B}$ (MML over $\mathcal{D}$, nonparametric naive estimate)
      - How do we enforce the constraint
        - Distribution constrain: restrict $\pi_{\theta}$ distrib to match ${\pi_B}$
        - Support constraints only restrict theactions selected from ${\pi_{\theta}}$ to be within the support of the actions selected by ${\pi_B}$, but not the probabilities
  - Importance sampling - reweight ${\pi_B}$ samples to evaluate $\pi_{\theta}$ (eq. 8)
    - $\nabla_{\theta} J(\theta) = \mathbb{E}_{\tau \sim p_{\pi_{\beta}}(\cdot)} \left[ w_{0:H} \sum_{t=0}^{H} \nabla_{\theta} \gamma^{t} \log \pi_{\theta}(a_{t}|s_{t}) \hat{Q}(s_{t}, a_{t}) \right]$
    - Requirement: $\hat{\pi}_{\beta}$ estimate, weights $w_{i:j} = \prod_{t=i}^{j} \pi_{\theta}(a_{t}|s_{t}) / \hat{\pi}_{\beta}(a_{t}|s_{t})$
    - Disadvantages: $w_{0:H}$ exponential in $H$ -> high variance, poorly conditioned
    - Variations (variance reduction)
      - Per-decision $w_{t:t'}$ in place of $w_{t:H}$, unbiased (eq. 21)
      - Self-normalized $\tilde{w}_{i:j} = w_{i:j} / \mathbb{E}_{\tau \sim p_{\pi_{\beta}}}[w_{i:j}]$ - trades bias for large variance drop
      - Doubly robust: $\hat{Q}$ as control variate, unbiased if either ${\pi_B}$ known or $\hat{Q}$ correct (eq. 22)
      - Marginalized IS: state-marginal ratio $\rho^{\pi}(s) = d^{\pi}(s) / d^{\pi_{\beta}}(s)$, avoids the $O(H)$ product (eq. 23)
  - Regularization - penalty term $R$ to impose behaviors that do **not** depend on ${\pi_B}$
    - Policy reg (maximize, eq. 9): $J(\theta) = \mathbb{E}_{s,a \sim \mathcal{D}}[Q^{\pi_{\theta}}(s,a) + R(\theta)]$
    - Value reg (minimize, eq. 10): $J(\phi) = \mathbb{E}_{s,a,s' \sim \mathcal{D}}\left[ \left( r(s,a) + \gamma \mathbb{E}_{a' \sim \pi_{off}(\cdot|s)}[Q^{\pi_{\phi}}(s',a')] - Q^{\pi_{\phi}}(s,a) \right)^{2} \right] + R(\phi)$
    - Requirement: none on ${\pi_B}$ - no $\hat{\pi}_{\beta}$ estimate needed
    - Advantage: less conservative than policy constraints, policy not limited to ${\pi_B}$
    - Disadvantages: does not prevent OOD actions on its own -> usually paired w/ policy constraints or conservative models
    - Variations
      - Policy - entropy (SAC, eq. 25): $R(\theta) = \mathbb{E}_{s \sim \mathcal{D}}[\mathcal{H}(\pi_{\theta}(\cdot|s_{t}))] = -\mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_{\theta}(\cdot|s)}[\log \pi_{\theta}(a|s)]$ - controls stochasticity, avoids premature collapse of policy variance
      - Value - push Q down for $a \sim \pi_{\theta}$ (eq. 26): $R(\phi) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_{\theta}(\cdot|s)}[Q^{\pi_{\phi}}(s,a)]$
      - Value - CQL (eq. 27): $R(\phi) = \max_{\mu} \mathbb{E}_{s \sim \mathcal{D}, a \sim \mu(\cdot|s)}[Q^{\pi_{\phi}}(s,a)] - \mathbb{E}_{s \sim \mathcal{D}, a \sim \hat{\pi}_{\beta}(\cdot|s)}[Q^{\pi_{\phi}}(s,a)] + R(\mu)$ - learns a lower bound of true $Q$; push OOD down, in-distribution up
        - Drawback: saddle point (min over $\phi$, max over $\mu$) -> unstable
        - With $R(\mu) = \mathcal{H}(\mu)$ the inner max solves to $\mu^{*} = \frac{1}{Z}\exp(Q^{\pi}(s,a))$ -> logsumexp form, no saddle point
  - Uncertainty estimation - penalize uncertainty of the approximation, relax the constraint where uncertainty is low (eq. 11)
    - $J(\theta) = \mathbb{E}_{s,a \sim \mathcal{D}}\left[ \mathbb{E}_{Q^{\pi} \sim P_{\mathcal{D}}(\cdot)}[Q^{\pi}(s,a)] - \alpha U_{P_{\mathcal{D}}}(P_{\mathcal{D}}(\cdot)) \right]$
    - Requirement: a distribution over Q-functions $P_{\mathcal{D}}(Q^{\pi})$ + an uncertainty estimator $U_{P_{\mathcal{D}}}(\cdot)$ for it
    - Advantage: dial between conservative and naive off-policy RL by how much we trust the model's generalization
    - Disadvantages: ensembles give little diversity in practice -> uncertainty **under**estimated -> still prone to OOD actions
    - Variations (mostly about how to define $P_{\mathcal{D}}$ and $U_{P_{\mathcal{D}}}$)
      - Naive ensemble (eq. 29): $K$ Q-functions trained independently on disjoint partitions of $\mathcal{D}$, $P_{\mathcal{D}}(Q^{\pi}) \approx \frac{1}{K}\sum_{i=1}^{K} \delta(Q^{\pi} = Q^{\pi}_{\phi_{i}})$, then $U_{P_{\mathcal{D}}} = \mathrm{Var}$ (sample mean / variance)
      - REM (eq. 30): random convex combination $Q^{\pi}(s,a) = \sum_{i=1}^{K} \alpha_{i} Q^{\pi}_{\phi_{i}}(s,a)$ s.t. $\sum_{i}\alpha_{i} = 1, \forall i \, \alpha_{i} > 0$ - good on high-coverage datasets, still lags policy constraints
- Other modifications (not loss terms - survey III-E..III-H)
  - Model-based (III-E) - estimate dynamics $T_{\psi_T}(s_{t+1}|s_t, a_t)$ and reward $r_{\psi_r}(s_t, a_t)$ by supervised regression on $\mathcal{D}$, then use as a proxy env to simulate transitions and plan
    - Works well when data coverage is high (easy to learn an accurate model)
    - Problem: an offline model cannot correct its mistakes by interacting -> model distributional shift
    - Fix: conservative model - penalize the reward in OOD states (eq. 12) $\tilde{r}_{\psi_r}(s,a) = r_{\psi_r}(s,a) - \lambda U_r(s,a)$, with $U_r$ low on $\mathcal{D}$ and high elsewhere (reuses the uncertainty estimators above)
  - One-step (III-F) - one policy evaluation step, then one policy improvement step
    - Terminology: *iteration* = a single update; *step* = many iterations until the quantity converges
    - Multi-step: alternate eval/improvement until $Q^{\pi_{off}}$ converges -> targets use actions from $\pi_{off}$ while $Q$ is trained on ${\pi_B}$ actions -> distributional shift
    - One-step: multiple state sweeps to get an accurate $Q^{\pi_{\beta}}$, then a single improvement step to find the best policy
    - Advantage: policy evaluation never sees OOD actions -> no constraint terms needed in the loss at all
  - Imitation learning (III-G) - mimic ${\pi_B}$, BC at its core
    - Plain BC (eq. 13): minimize $J(\theta) = D(\pi_{\beta}(\cdot|s), \pi_{\theta}(\cdot|s))$, $D$ an $f$-divergence (e.g. cross-entropy)
      - Subject to distributional shift: tiny mistakes -> unseen states -> compounding errors
      - Only succeeds if the data is mostly expert behavior
    - Filtering: use value functions / heuristics to keep only the good trajectories, then BC on those
    - Conditional policy (eq. 14): learn $\pi_{\theta}(a_t|s_t, \omega)$ where the outcome $\omega \sim g(\cdot|\tau_{t:H})$ is conditioned on the remaining trajectory
      - $J(\theta) = \mathbb{E}_{\tau \sim p_{\pi_{\beta}}(\cdot), \, t \sim U(1,H), \, \omega \sim g(\cdot|\tau_{t:H})}\left[ \log \pi_{\theta}(a_t|s_t, \omega) \right]$
      - Key challenge: defining the outcome function $g$ - varies by data and task
  - Trajectory optimization (III-H) - learn a joint state-action model over entire trajectories (eq. 15) $p_{\pi_{\beta}}(\tau) = p_{\pi_{\beta}}(s_0, a_0, \dots, s_H)$
    - With a good model, plan an optimal set of actions from the initial state $s_0$
    - Less prone to OOD actions: state/action anchors throughout the trajectory prevent deviating far from ${\pi_B}$
    - Large sequence models (transformers) suit the offline setting - no active data collection, update between trials

# Algorithmic Review

Methods per modification type (survey Sec. IV, Table II). No checkmarks in Table II = vanilla model-free multistep actor-critic.

## Policy constraints (IV-A)

- **BCQ** [14] - seminal offline RL method, *direct* constraint. $\pi_{\theta}(a,s) = \arg\max_{a_i + \xi_{\theta}(s,a_i)} Q^{\pi_{\phi}}(s, a_i + \xi_{\theta}(s,a_i))$, $a_i \sim \hat{\pi}_{\beta}$ (eq. 16)
  - $\hat{\pi}_{\beta}$ = parametric generative model; $\xi_{\theta}$ = perturbation model bounded to $[-\Phi, \Phi]$; $N$ samples
  - $N=1, \Phi=0 \Rightarrow \pi_{\theta} = \pi_{\beta}$; $N \to \infty, \Phi = a_{max} \Rightarrow$ unconstrained (online RL)
  - Struggles to filter out poor behaviors
- **BEAR** [15] - *indirect*, support matching instead of distribution matching
  - MMD divergence $D_{MMD}$ w/ Gaussian kernel; sampled MMD over few samples $\approx$ MMD to $\pi_{\beta}$'s support
  - Argument: distribution matching stops us exploiting good actions and ignoring poor ones
  - Finds good policies even when ${\pi_B}$ is a mix of suboptimal behaviors
- **BRAC** [16] - *direct*, general framework for behavior-regularized actor-critic
  - Subtract a divergence from either policy improvement (BRAC-p) or policy evaluation (BRAC-v); both use $D_{KL}(\pi_{\theta} \Vert \pi_{\beta})$
  - **Key finding**: BCQ/BEAR's design choices (MMD vs KL etc.) matter far less than hyperparameter search; value penalty > policy penalty
- **Fisher-BRC** [38] - Fisher divergence $D_F(\pi_{\theta} \Vert \pi_{\beta})$ on the entropy-regularized policy
  - Entropy-smoothed Q: $Q^{\pi_{\phi}}(s,a) = O^{\pi}_{\phi}(s,a) + \log \hat{\pi}_{\beta}(s,a)$, $O^{\pi}_{\phi}$ = state-action offset function
  - Gradient penalty $\Vert \nabla_a O^{\pi}_{\phi}(s,a) \Vert^{2}$ so $O^{\pi}_{\phi}$ doesn't dominate $\hat{\pi}_{\beta}$
  - Can also be read as a value regularization method
- **AWR** [36] (analogous to MARWIL [49]) - *implicit* KL constraint, no $\hat{\pi}_{\beta}$ needed
  - $V^{\pi}_{\phi}$ fit by supervised regression on Monte Carlo rollouts
  - MC advantage as exponential weight in weighted max-log-likelihood (eq. 7): $A^{\pi}_{MC}(s_t,a_t) = R_t - V^{\pi}_{\phi}(s_t)$ (eq. 17)
- **AWAC** [39] - AWR but with a Q-function for the advantage -> lower variance, better sample efficiency
  - $A^{\pi}_{AWAC}(s,a) = Q^{\pi}_{\phi}(s,a) - \mathbb{E}_{\tilde{a} \sim \pi_{\theta}}[Q^{\pi}_{\phi}(s,\tilde{a})]$ (eq. 19)
  - $Q$ fit by bootstrapped regression w/ $\mathbb{E}_{a' \sim \pi_{\theta_{k-1}}}$ instead of $\max$ (eq. 18)
  - Action-dependent baseline introduces bias (cf. Q-Prop [50]); left uncorrected
- **TD3+BC** [44] - TD3 [51] + a BC regularizer, *implicit* constraint
  - $J(\theta) = \mathbb{E}_{s,a \sim \mathcal{D}, \tilde{a} \sim \pi_{\theta}}[\lambda Q^{\pi_{\phi}}(s,\tilde{a}) - (\tilde{a}-a)^{2}]$ (eq. 20), $\lambda$ = regularizer strength
  - Plus z-score normalization of states; competitive w/ SOTA on D4RL Gym-MuJoCo
  - **Takeaway: performance is often not tied to algorithmic complexity**
- Implicit methods (BEAR, AWR, AWAC, TD3+BC) are best for **online fine-tuning** - direct methods must re-estimate $\hat{\pi}_{\beta}$ every time $\mathcal{D}$ grows

## Importance sampling (IV-B)

- **Per-decision IS** [40] - $w_{0:t-1}$ per reward instead of $w_{0:H}$; unbiased, lower variance (eq. 21)
- **Self-normalized weights** [40] - $\tilde{w}_{i:j}$; adds bias, large variance reduction
- **Doubly robust** [41] - $\hat{Q}$ as control variate (eq. 22); unbiased if *either* ${\pi_B}$ known *or* $\hat{Q}$ correct
- **Marginalized IS** [52] - state-marginal ratio $\rho^{\pi}(s)$, avoids the $O(H)$ product; satisfies a Bellman eq. (eq. 23)

## Regularization (IV-C)

- **SAC entropy** [54] - policy reg, $R(\theta) = \mathbb{E}_{s \sim \mathcal{D}}[\mathcal{H}(\pi_{\theta}(\cdot|s_t))]$ (eq. 25); controls stochasticity
- **Nachum et al.** [25] - value reg, push $Q$ down for $a \sim \pi_{\theta}$ (eq. 26)
- **CQL** [17] - value reg, learns a **lower bound** of the true $Q$ (eq. 27)
  - Saddle point (min over $\phi$, max over $\mu$) -> unstable
  - $R(\mu) = \mathcal{H}(\mu) \Rightarrow \mu^{*} = \frac{1}{Z}\exp(Q^{\pi})$ -> logsumexp form, no saddle point

## Uncertainty estimation (IV-D)

- **Naive ensemble** [57] - $K$ Q-functions on disjoint partitions, $U_{P_{\mathcal{D}}} = \mathrm{Var}$ (eq. 29); little diversity -> uncertainty underestimated
- **REM** [57] - random convex combination of Q-functions (eq. 30); good on high-coverage data, lags policy constraints

## Model-based (IV-E)

- **MOReL** [20] - epistemic uncertainty from an ensemble of dynamics models
  - Disagreement $dis_{i,j}(s_t,a_t) = \mathbb{E}[\Vert s^{i}_{t+1} - s^{j}_{t+1} \Vert]$ (eq. 31)
  - $U_r(s,a) = r_{max}$ if $\max_{i,j} dis_{i,j} >$ threshold, else $0$ (eq. 32) -> pessimistic MDP
- **MOPO** [42] - Gaussian ensemble $T_{\psi_i}(s'|s,a) = \mathcal{N}(\mu_i, \Sigma_i)$ (eq. 33)
  - $U_r(s,a) = \max_i \Vert \Sigma_i(s,a) \Vert_F$ (eq. 34) - max prediction uncertainty
- With $\tilde{r}$ in hand -> classic model-based approaches: Dyna-style sampling to train a model-free algo, or planning w/ LQR [59] / MCTS [60]
- **BREMEN** [43] - ensemble of models of the **behavior** MDP (not a pessimistic one) + trust-region updates (eq. 35)
  - Akin to *direct* policy constraint: $\hat{\pi}_{\beta}$ initializes $\pi_{\theta_0}$ (Gaussian, mean from $\hat{\pi}_{\beta}$, unit variance)
  - Main selling point is **deployment efficiency** - # of distinct data-collection policies; good policies from random init in 5-10 deployments
  - Caveat: no other work benchmarks deployment efficiency -> hard to compare
- **COMBO** [18] - model-based CQL; single dynamics model $T_{\psi_T}$ trained by max log-likelihood, inducing MDP $\hat{\mathcal{M}}$
  - Same as CQL evaluation but $\rho(s,a) = d^{\pi}_{\hat{\mathcal{M}}}(s)\pi(a|s)$ and $d^{\mu}_{f} = f d(s,a) + (1-f) d^{\mu}_{\hat{\mathcal{M}}}(s,a)$ (eqs. 36-38)
  - $f \in [0,1]$ tunes conservatism (larger $f$ = more offline data = more conservative)
  - Push $Q$ down on model rollouts, up on offline data; still a lower bound on true $Q$
  - **Advantage: removes the need for uncertainty quantification** (challenging + often unreliable); similar in spirit to adversarial training [62]

## One-step (IV-F)

- **One-step framework** [22] - single policy eval + single improvement, multiple improvement operators tested
  - Beats multistep on several Gym envs; success attributed to being able to use **very weak regularization** -> approximators fit the true $Q$ more freely
  - Multistep methods diverge at low regularization weight; best performance is at the smallest weight that does not diverge, more than that -> too conservative
  - **Underperforms multistep when $\mathcal{D}$ is largely suboptimal or has good state-action coverage** (there, iterative error exploitation hurts less)
  - Best config: standard Bellman error for eval + exponentiated advantage w/ max log-likelihood for improvement (eq. 7)
- **IQL** [23] - implicit Q-learning; contribution is in the policy evaluation step
  - Uses a learned $V^{\pi}_{\psi}$ as the target instead of sampling target actions: $J(\phi) = \mathbb{E}_{s,a,s' \sim \mathcal{D}}[r(s,a) + V^{\pi}_{\psi}(s') - Q^{\pi}_{\phi}(s,a)]$ (eq. 39)
  - $J(\psi) = \mathbb{E}_{s,a \sim \mathcal{D}}[\ell(V^{\pi}_{\psi}(s) - Q^{\pi}_{\phi}(s,a))]$ (eq. 40); MSE would give $V \to \mathbb{E}_{a \sim \pi_{\beta}}[Q]$, but we want Bellman **optimality**
  - Fix: **expectile regression** $\ell^{\tau}_{2}(x) = (1-\tau)x^{2}$ if $x>0$ else $\tau x^{2}$ (eq. 41) - penalizes negative errors more, so $V$ prefers to be larger than $Q$
  - Effectively $V^{\pi}(s) = \max_{a \in \Omega(s)} Q^{\pi}(s,a)$ where $\Omega(s) = \{a : \pi_{\beta}(a|s) \geq \epsilon\}$ (eq. 42) - best action *within data support*
  - Only trains on in-dataset $(s,a)$ -> no OOD overestimation; policy extraction via AWR-style exponentiated advantage
  - Unlike [22] it **does** perform iterative dynamic programming (optimality-based updates -> can improve on ${\pi_B}$)
  - One of the most successful methods on D4RL [6]; reliable from AntMaze to Adroit

## Imitation learning (IV-G)

- **BAIL** [21] - fit $V^{\pi}_{\phi}$ to the **upper envelope** of returns in $\mathcal{D}$, then BC on the best pairs
  - $J(\phi) = \mathbb{E}[\sum_t (V^{\pi}_{\phi}(s_t) - R_t)^{2} w(s_t)]$ w/ $w(s_t) = 1$ if $V^{\pi}_{\phi}(s_t) < R_t$ else $K$ (eqs. 43-44); $K \gg 1$ -> upper envelope
  - Keep pairs with $R_t > \rho V^{\pi}_{\phi}(s_t)$ (eq. 45), $\rho = 0.25$
- **ABM** [46] - advantage-weighted behavior model as a **prior**, then MPO [63]
  - $n$-step advantage $A^{\pi}_{n}$ (eq. 46); filter w/ an **indicator** instead of an exponential: $J(\theta_{abm}) = \mathbb{E}[\log \pi_{\theta_{abm}}(a|s) \mathbb{1}(\hat{A}^{\pi}(s,a) > 0)]$ (eq. 47)
  - Then MPO constrains $\pi_{\theta}$ to $\pi_{\theta_{abm}}$ by KL - better than KL to ${\pi_B}$ when ${\pi_B}$ is suboptimal
- **CRR** [45] - same indicator filtering, but a more **pessimistic** advantage $A^{\pi}_{CRR}(s,a) = Q^{\pi}_{\phi}(s,a) - \max_{\tilde{a}} Q^{\pi}_{\phi}(s,\tilde{a})$ (eq. 48)
  - $\max$ in place of AWAC's expectation; beats implicit policy constraints on mixed expert/suboptimal data (those are too permissive and copy inferior actions)
- **RvS** [47] - conditional BC (eq. 14) with two outcome choices
  - Goal-conditioned $g_g(\omega|\tau_{t:H}) = U(s_{t+1}, s_H)$ (eq. 49)
  - Reward-conditioned $g_r(\omega|\tau_{t:H}) = \mathbb{1}(\omega = \frac{1}{H-t+1}\sum_{t'} r(s_{t'},a_{t'}))$ (eq. 50)
  - Simple formulation + more network capacity/regularization matches or beats CQL in some domains
  - **Choice of conditioning variable is crucial**: $g_g$ wins where compositionality helps (AntMaze, FrankaKitchen), $g_r$ wins on Gym-MuJoCo (where $g_g$ doesn't apply)

## Trajectory optimization (IV-H)

- Both represent a trajectory as returns-to-go interleaved w/ states and actions: $\tau = (R_0, s_0, a_0, R_1, s_1, a_1, \dots, R_H, s_H)$ (eq. 51)
- **TT** (Trajectory Transformer) [28] - max log-likelihood of **every** token conditioned on previous ones; plan w/ **beam search** [64] + reward-to-go estimates
- **DT** (Decision Transformer) [27] - MSE on **predicted actions only** (states + returns-to-go deemed unnecessary); no planning - condition rollouts on a target return, then subtract observed reward to update the reward-to-go
- Expensive to train, but strong in **sparse reward** settings where TD methods fail (those need dense rewards to propagate Q-values over long horizons)
