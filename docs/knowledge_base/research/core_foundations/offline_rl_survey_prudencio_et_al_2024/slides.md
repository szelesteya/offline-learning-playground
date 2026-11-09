---
marp: true
theme: default
paginate: true
math: katex
size: 16:9
style: |
  section {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    font-size: 24px;
    padding: 35px 50px;
  }
  h1 {
    color: #1a56db;
  }
  h2 {
    color: #2b3a4a;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 6px;
    margin-bottom: 18px;
  }
  h3 {
    color: #1e40af;
    margin-top: 8px;
    margin-bottom: 6px;
  }
  footer {
    font-size: 14px;
    color: #718096;
  }
  ul {
    margin-top: 4px;
    margin-bottom: 8px;
  }
  li {
    margin-bottom: 3px;
  }
  code {
    background-color: #f1f5f9;
    padding: 2px 5px;
    border-radius: 4px;
    font-size: 0.9em;
  }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# A Survey on Offline Reinforcement Learning
### Taxonomy, Review, and Open Problems

**Rafael Figueiredo Prudencio, Marcos R. O. A. Maximo, Esther Luna Colombini** (2024)
*IEEE Transactions on Neural Networks and Learning Systems*

[[Prudencio et al., 2024]](#ref-prudencioSurveyOfflineReinforcement2024)

---

## 1. Introduction: The Offline RL Paradigm

- **Online RL**: Agent interacts actively with the environment; updates policy and collects fresh rollouts.
- **Off-Policy RL**: Reuses historical replay buffer data, but still requires continuous online environment interaction.
- **Offline RL (Batch RL)**: Learns strictly from a static dataset $\mathcal{D}$ collected by an unknown behavior policy $\pi_{\beta}$ with **zero environment interaction during training**.

<div style="display: flex; gap: 20px; margin-top: 10px;">
<div style="flex: 1;">

### Why Offline RL?
- Eliminates real-world interaction risks and costs (healthcare, robotics, autonomous driving, energy).
- Capitalizes on vast historical and diverse datasets.

</div>
<div style="flex: 1;">

### Desirable RL Properties
- **Generalization**: Broad data coverage.
- **Filtering**: Extracting good policies from noisy data.
- **Compositionality**: Stitching partial good trajectories.

</div>
</div>

<!-- _footer: "Ref: [[Prudencio et al., 2024]](#ref-prudencioSurveyOfflineReinforcement2024)" -->

---

## 2. Problem Formalization & The Core Challenge

- **Markov Decision Process**: $\mathcal{M} = (\mathcal{S}, \mathcal{A}, T, d_0, r, \gamma)$
- **Static Dataset**: $\mathcal{D} = \{(s_t, a_t, s_{t+1}, r_t)\}_{i=1}^N$ with $a_t \sim \pi_{\beta}(\cdot|s_t)$ and $s_t \sim d^{\pi_{\beta}}(\cdot)$.
- **Objective**: Maximize return $J(\pi) = \mathbb{E}_{\tau \sim p_{\pi}}[R_{0:H}]$.

### The Fundamental Obstacle: Distributional Shift
When minimizing the Bellman error on offline data:
$$J(\phi) = \mathbb{E}_{(s,a,s') \sim \mathcal{D}}\left[ \left( r(s,a) + \gamma \mathbb{E}_{a' \sim \pi_{\text{off}}(\cdot|s')}[Q^{\pi}_{\phi}(s', a')] - Q^{\pi}_{\phi}(s, a) \right)^2 \right]$$
- Target action $a' \sim \pi_{\text{off}}$ differs from dataset action $a \sim \pi_{\beta}$.
- Unseen/out-of-distribution (OOD) actions suffer from **catastrophic value overestimation**.
- Without online error correction, sampling and function approximation errors compound uncontrollably.

<!-- _footer: "Ref: [[Prudencio et al., 2024]](#ref-prudencioSurveyOfflineReinforcement2024)" -->

---

## 3. Taxonomy: Algorithmic Structure & Modifiers

The survey proposes a unified categorization distinguishing **is-a** and **has-a** relationships:

<div style="display: flex; gap: 24px;">
<div style="flex: 1;">

### Architectural Classes (*Is-A*)
1. **Model-Free Multistep**: Alternating policy evaluation & improvement.
2. **Model-Based**: Learn proxy dynamics $\hat{T}(s'|s,a)$ for synthetic rollouts or planning.
3. **One-Step RL**: Decouple evaluation from improvement (single state sweep).
4. **Imitation Learning**: Behavioral cloning + trajectory filtering or conditioning.
5. **Trajectory Optimization**: Autoregressive sequence models (Transformers) over full trajectories.

</div>
<div style="flex: 1;">

### Optional Loss Terms (*Has-A*)
- **Policy Constraints**: Direct or implicit divergences ($D_{KL}, D_{\text{MMD}}, D_F$).
- **Value Regularization**: Pushing down OOD values (CQL).
- **Uncertainty Estimation**: Epistemic penalty via model or $Q$ ensembles.
- **Importance Sampling**: Reweighting trajectory transitions ($\rho^{\pi}$).

</div>
</div>

<!-- _footer: "Ref: [[Prudencio et al., 2024]](#ref-prudencioSurveyOfflineReinforcement2024)" -->

---

## 4. Loss Term Formulations

### Policy Constraints
- **Direct** ($\hat{\pi}_{\beta}$ estimated): $\max_{\theta} \mathbb{E}[Q(s, a)] \quad \text{s.t.} \quad D(\pi_{\theta}(\cdot|s), \hat{\pi}_{\beta}(\cdot|s)) \leq \epsilon$
- **Implicit** (KKT Lagrangian solution): $\max_{\theta} \mathbb{E}_{(s,a) \sim \mathcal{D}}\left[ \log \pi_{\theta}(a|s) \exp\left(\frac{1}{\lambda} \hat{A}^{\pi}(s, a)\right) \right]$

### Value & Policy Regularization
- **Policy Entropy**: $\mathcal{R}(\theta) = -\mathbb{E}[\log \pi_{\theta}(a|s)]$ [[Haarnoja et al., 2018]](#ref-haarnojaSoftActorcriticOffpolicy2018).
- **Value Penalty (CQL)**: $\min_Q \alpha \mathbb{E}_s \left[ \log \sum_a \exp(Q(s, a)) - \mathbb{E}_{a \sim \hat{\pi}_{\beta}}[Q(s,a)] \right]$ [[Kumar et al., 2020]](#ref-kumarConservativeQLearningOffline2020).

### Importance Sampling & Uncertainty
- Variance reduction via per-decision weights [[Precup et al., 2000]](#ref-precupEligibilityTracesOffpolicy2000), Doubly Robust estimators [[Jiang & Li, 2016]](#ref-jiangDoublyRobustOffpolicy2016), or Marginalized IS [[Hallak & Mannor, 2017]](#ref-hallakConsistentOnlineOffpolicy2017).
- Ensemble variance penalty: $J(\theta) = \mathbb{E}[Q(s,a) - \alpha U(s,a)]$ [[Agarwal et al., 2020]](#ref-agarwalOptimisticPerspectiveOffline2020).

<!-- _footer: "Ref: [[Kumar et al., 2020]](#ref-kumarConservativeQLearningOffline2020); [[Haarnoja et al., 2018]](#ref-haarnojaSoftActorcriticOffpolicy2018)" -->

---

## 5. Algorithmic Review: Policy Constraint Methods

- **BCQ** [14] [[Fujimoto et al., 2018]](#ref-fujimotoOffPolicyDeepReinforcement2018):
  - Parametric generative model $\hat{\pi}_{\beta}$ + perturbation model $\xi_{\theta}(s, a) \in [-\Phi, \Phi]$. Restricts actions to the vicinity of the behavior policy.
- **BEAR** [15] [[Kumar et al., 2019]](#ref-kumarStabilizingOffPolicyQLearning2019):
  - Support matching over distribution matching using sampled MMD with Gaussian kernels. Avoids copying suboptimal behavior probabilities.
- **BRAC** [16] [[Wu et al., 2019]](#ref-wuBehaviorRegularizedOffline2019):
  - Comprehensive framework showing that hyperparameter selection and value penalties dominate choice of divergence ($D_{KL}$ vs $D_{\text{MMD}}$).
- **Fisher-BRC** [38] [[Kostrikov et al., 2021a]](#ref-kostrikovOfflineReinforcementLearning2021a):
  - Constrains entropy-regularized policy using Fisher divergence $D_F(\pi_{\theta} \Vert \pi_{\beta})$ with critic gradient penalties.

<!-- _footer: "Ref: [[Fujimoto et al., 2018]](#ref-fujimotoOffPolicyDeepReinforcement2018); [[Kumar et al., 2019]](#ref-kumarStabilizingOffPolicyQLearning2019); [[Wu et al., 2019]](#ref-wuBehaviorRegularizedOffline2019)" -->

---

## 6. Algorithmic Review: Implicit Policy Constraints

Implicit methods avoid explicit estimation of behavior density $\hat{\pi}_{\beta}$:

- **AWR** [36] [[Peng et al., 2019]](#ref-pengAdvantageWeightedRegressionSimple2019) / **MARWIL** [49] [[Wang et al., 2018]](#ref-wangExponentiallyWeightedImitation2018):
  - Evaluates $V(s)$ via supervised regression on Monte Carlo returns; extracts policy with exponentiated advantage weights $\exp(A_{\text{MC}} / \beta)$.
- **AWAC** [39] [[Nair et al., 2021]](#ref-nairAWACAcceleratingOnline2021):
  - Extends AWR by using TD-learned $Q$-functions for lower-variance advantage computation: $A(s, a) = Q(s, a) - \mathbb{E}_{\tilde{a}}[Q(s, \tilde{a})]$. Excellent for online fine-tuning.
- **TD3+BC** [44] [[Fujimoto & Gu, 2021]](#ref-fujimotoMinimalistApproachOffline2021):
  - Minimalist approach: vanilla TD3 [[Fujimoto et al., 2018]](#ref-fujimotoAddressingFunctionApproximation2018) actor loss with normalized states and a BC regularizer:
  $$J(\theta) = \mathbb{E}_{(s,a) \sim \mathcal{D}}[\lambda Q(s, \pi_{\theta}(s)) - (\pi_{\theta}(s) - a)^2]$$
  - Proves that simplicity often rivals complex generative models on D4RL Gym-MuJoCo [[Fu et al., 2020]](#ref-fuD4RLDatasetsDeep2020).

<!-- _footer: "Ref: [[Peng et al., 2019]](#ref-pengAdvantageWeightedRegressionSimple2019); [[Nair et al., 2021]](#ref-nairAWACAcceleratingOnline2021); [[Fujimoto & Gu, 2021]](#ref-fujimotoMinimalistApproachOffline2021)" -->

---

## 7. Algorithmic Review: Model-Based Offline RL

- **MOReL** [20] [[Kidambi et al., 2021]](#ref-kidambiMOReLModelBasedOffline2021):
  - Evaluates dynamics disagreement $\text{dis}_{i,j}(s, a) = \mathbb{E}[\|s'_{i} - s'_{j}\|]$.
  - Replaces reward with $r_{\max}$ penalty on high disagreement $\implies$ pessimistic MDP.
- **MOPO** [42] [[Yu et al., 2020]](#ref-yuMOPOModelbasedOffline2020):
  - Maximum prediction variance across Gaussian dynamics ensemble: $\tilde{r} = r - \lambda \max_i \|\Sigma_i(s, a)\|_F$.
- **BREMEN** [43] [[Matsushima et al., 2020]](#ref-matsushimaDeploymentEfficientReinforcementLearning2020):
  - Optimizes deployment efficiency (5–10 iterations) using trust-region updates over behavioral model ensembles.
- **COMBO** [18] [[Yu et al., 2022]](#ref-yuCOMBOConservativeOffline2022):
  - Model-based CQL. Pushes down $Q$-values on synthetic model rollouts and pushes up $Q$-values on offline dataset transitions, eliminating explicit uncertainty estimation.

<!-- _footer: "Ref: [[Kidambi et al., 2021]](#ref-kidambiMOReLModelBasedOffline2021); [[Yu et al., 2020]](#ref-yuMOPOModelbasedOffline2020); [[Yu et al., 2022]](#ref-yuCOMBOConservativeOffline2022)" -->

---

## 8. Algorithmic Review: One-Step & Imitation Learning

- **One-Step Framework** [22] [[Brandfonbrener et al., 2021]](#ref-brandfonbrenerOfflineRLOffPolicy2021):
  - One policy evaluation step followed by a single improvement step. Prevents error accumulation; functions with very weak regularization.
- **Implicit Q-Learning (IQL)** [23] [[Kostrikov et al., 2021b]](#ref-kostrikovOfflineReinforcementLearning2021):
  - Approximates $V(s) \approx \max_{a \in \text{supp}(\mathcal{D})} Q(s, a)$ via **expectile regression**:
  $$L_2^{\tau}(u) = |\tau - \mathbb{I}(u < 0)| u^2$$
  - Eliminates OOD action queries entirely while retaining dynamic programming optimality.
- **Filtered & Conditional BC**:
  - **BAIL** [21] [[Chen et al., 2020]](#ref-chenBAILBestActionImitation2020): Fits value upper envelope; clones best-action subset ($R > \rho V$).
  - **CRR** [45] [[Wang et al., 2021]](#ref-wangCriticRegularizedRegression2021) & **ABM** [46] [[Siegel et al., 2020]](#ref-siegelKeepDoingWhat2020): Advantage indicator filtering.
  - **RvS** [47] [[Emmons et al., 2022]](#ref-emmonsRvSWhatEssential2022): Conditional supervised learning on goals $g_g$ or returns $g_r$.

<!-- _footer: "Ref: [[Brandfonbrener et al., 2021]](#ref-brandfonbrenerOfflineRLOffPolicy2021); [[Kostrikov et al., 2021]](#ref-kostrikovOfflineReinforcementLearning2021)" -->

---

## 9. Algorithmic Review: Trajectory Optimization

<div style="display: flex; gap: 24px;">
<div style="flex: 1;">

### Decision Transformer (DT) [27] [[Chen et al., 2021]](#ref-chenDecisionTransformerReinforcement2021)
- Represents trajectory as token sequence:
  $$\tau = (\hat{R}_1, s_1, a_1, \dots, \hat{R}_T, s_T, a_T)$$
- Supervised autoregressive action prediction conditioned on target return-to-go $\hat{R}$.
- Simple MSE loss on actions without predicting future dynamics or TD backups.

</div>
<div style="flex: 1;">

### Trajectory Transformer (TT) [28] [[Janner et al., 2021]](#ref-jannerOfflineReinforcementLearning2021)
- Full joint generative sequence model predicting discretized $(s, a, r)$ tokens via maximum log-likelihood.
- Trajectory rollouts evaluated and planned via **beam search** guided by $Q$-functions (e.g. from IQL).
- Outstanding performance on sparse-reward tasks.

</div>
</div>

<!-- _footer: "Ref: [[Chen et al., 2021]](#ref-chenDecisionTransformerReinforcement2021); [[Janner et al., 2021]](#ref-jannerOfflineReinforcementLearning2021)" -->

---

## 10. Off-Policy Evaluation (OPE): Core Approaches

OPE evaluates policy expected return $\hat{J}(\pi)$ using offline data $\mathcal{D}_e$ without environment interaction:

- **1. Model-Based OPE**: Evaluate expected return under fitted dynamics model $\hat{T}_{\psi_T}$:
  $$\hat{J}(\pi) = \mathbb{E}_{\tau \sim p_{\psi_T}}\left[\sum_{t=0}^H \gamma^t r(s_t, a_t)\right]$$
- **2. Importance Sampling (IS)**: Reweight transitions via cumulative density ratios $w_{0:H}$; variance reduced using Doubly Robust estimators [[Jiang & Li, 2016]](#ref-jiangDoublyRobustOffpolicy2016) or Marginalized IS [[Hallak & Mannor, 2017]](#ref-hallakConsistentOnlineOffpolicy2017).
- **3. Fitted Q-Evaluation (FQE)**: Train $Q^{\pi}$ by minimizing Bellman error on $\mathcal{D}_e$ under target policy $\pi$; evaluate $\hat{J}(\pi) = \mathbb{E}_{(s,a) \sim \mathcal{D}_e}[Q^{\pi}(s, a)]$.

---

## 11. OPE: Empirical Findings & Benchmarks

Evaluating and selecting policies offline for real-world deployment (Section V & VI-B):

<div style="display: flex; gap: 24px; margin-top: 10px;">
<div style="flex: 1;">

### FQE Dominance
- **Consistent SOTA**: Studies by Voloshin et al. [[Voloshin et al., 2019]](#ref-voloshinEmpiricalStudyOffpolicy2019) and Paine et al. [[Paine et al., 2020]](#ref-paineHyperparameterSelectionOffline2020) demonstrate that FQE consistently outperforms complex IS and DICE estimators (e.g., AlgaeDICE [[Nachum et al., 2019]](#ref-nachumAlgaeDICEPolicyGradient2019)).
- **DOPE Benchmark**: In the deep OPE benchmark [[Fu et al., 2021]](#ref-fuBenchmarksDeepOffpolicy2021), FQE achieves top overall performance across D4RL and RL Unplugged.

</div>
<div style="flex: 1;">

### Evaluation vs. Selection
- **Ranking over Absolute Values**: Estimating true return $V^{\pi}$ accurately is difficult, but ranking candidate policies (**Regret@k, Rank Correlation**) provides reliable hyperparameter tuning.
- **Early Stopping**: OPE ranking metrics allow early termination of training before policies overfit to static transitions.

</div>
</div>

<!-- _footer: "Ref: [[Voloshin et al., 2019]](#ref-voloshinEmpiricalStudyOffpolicy2019); [[Fu et al., 2021]](#ref-fuBenchmarksDeepOffpolicy2021); [[Paine et al., 2020]](#ref-paineHyperparameterSelectionOffline2020)" -->

---

## 12. Benchmark Dataset Design Factors

The survey identifies 8 critical properties for meaningful offline evaluations (Section VI-A1):

<div style="display: flex; gap: 24px; margin-top: 10px;">
<div style="flex: 1;">

### Behavioral Properties
1. **Narrow & Biased (NB)**: Concentrated demos; risk of OOD divergence.
2. **Undirected & Multitask (UM)**: Requires stitching disjoint fragments.
3. **Sparse Rewards (SR)**: Difficult long-horizon credit assignment.
4. **Suboptimal Data (SD)**: Demands policy improvement beyond BC.

</div>
<div style="flex: 1;">

### Realism & Distribution Properties
5. **Nonrepresentable (NR)**: True policy outside model capacity.
6. **Non-Markovian (NM)**: Complex human/heuristic policies.
7. **Realistic Domains (RD)**: High-fidelity noisy sensors & actuators.
8. **Nonstationarity (NS)**: Temporal parameter drift & wear.

</div>
</div>

<!-- _footer: "Ref: [[Fu et al., 2020]](#ref-fuD4RLDatasetsDeep2020); [[Gulcehre et al., 2020]](#ref-gulcehreRLUnpluggedSuite2020)" -->

---

## 13. Benchmark Suites: D4RL vs. RL Unplugged

<div style="display: flex; gap: 24px;">
<div style="flex: 1;">

### D4RL [[Fu et al., 2020]](#ref-fuD4RLDatasetsDeep2020)
- **Environments**: Gym-MuJoCo, AntMaze, Adroit dexterous hand, FrankaKitchen, CARLA, Flow.
- **Data Generation**: Hand-engineered controllers, human demos, random exploration.
- **Stitching**: Critical for maze navigation and kitchen tasks.

</div>
<div style="flex: 1;">

### RL Unplugged [[Gulcehre et al., 2020]](#ref-gulcehreRLUnpluggedSuite2020)
- **Environments**: DeepMind Control, Locomotion, ALE Atari, Real-World RL suite.
- **Data Generation**: Replay buffers of agents trained online.
- **Evaluation**: Explicit split between online and offline validation protocols.

</div>
</div>

---

## 14. Benchmark Performance Insights

Aggregated empirical findings across D4RL properties (Fig. 5):

<div style="display: flex; gap: 24px; margin-top: 8px;">
<div style="flex: 1;">

### Class Strengths & Standouts
- **Overall Leaders**: **Trajectory Transformer (TT)** [[Janner et al., 2021]](#ref-jannerOfflineReinforcementLearning2021) and **IQL** [[Kostrikov et al., 2021b]](#ref-kostrikovOfflineReinforcementLearning2021).
- **Trajectory Optimization**: Excels on sparse rewards (SR) and undirected multitask data (UM).
- **One-Step RL**: Superior stability on suboptimal data (SD) by avoiding iterative error compounding.

</div>
<div style="flex: 1;">

### Critical Benchmark Gaps
- **Stochastic Dynamics**: Severely underrepresented outside Atari.
- **Nonstationary Systems**: Only evaluated in the Real-World suite.
- **Multi-Agent Offline RL**: Completely absent from existing standardized benchmarks.

</div>
</div>

<!-- _footer: "Ref: [[Prudencio et al., 2024]](#ref-prudencioSurveyOfflineReinforcement2024)" -->

---

## 15. Practical Real-World Applications

Offline RL enables learning policies where online experimentation is prohibited:

<div style="display: flex; gap: 20px;">
<div style="flex: 1;">

### Healthcare [[Emerson et al., 2022]](#ref-emersonOfflineReinforcementLearning2022)
- Safe blood glucose regulation and insulin dosing for Type-1 diabetes patients.
- Online RL exploration is unsafe; offline RL maintains glucose strictly within clinical thresholds.

### Energy Management [[Zhan et al., 2022]](#ref-zhanDeepThermalCombustionOptimization2022)
- Combustion optimization for Thermal Power Generating Units (TPGUs).
- Combined historical plant logs and simulation to achieve safety-constrained efficiency far surpassing BC.

</div>
<div style="flex: 1;">

### Dialog Systems [[Verma et al., 2022]](#ref-vermaCHAIChatbotAITaskoriented2022)
- **CHAI (Chatbot AI)**: Learns goal-directed conversational policies from vast unlabeled chat logs without expensive human interaction loops.

### Robotics & Autonomous Driving
- Learning dexterous manipulation and highway navigation from human driver logs and industrial robot telemetry.

</div>
</div>

<!-- _footer: "Ref: [[Emerson et al., 2022]](#ref-emersonOfflineReinforcementLearning2022); [[Zhan et al., 2022]](#ref-zhanDeepThermalCombustionOptimization2022)" -->

---

## 16. Open Problems & Future Research Directions

1. **Reliable Hyperparameter Tuning & OPE**:
   - Inability to evaluate policies offline remains the #1 bottleneck for real-world deployment; current methods still lean on simulator rollouts.
2. **Incremental & Nonstationary Offline RL**:
   - Seamless offline-to-online fine-tuning without catastrophic forgetting or distribution collapse.
3. **Safety-Critical Offline RL**:
   - Preventing dangerous in-distribution actions via risk-averse objectives like Conditional Value-at-Risk (CVaR) [[Chow & Ghavamzadeh, 2014]](#ref-chowAlgorithmsCVaROptimization2014).
4. **Unsupervised Offline RL & Unlabeled Data**:
   - Leveraging vast reward-free transition datasets combined with sparse task rewards via reward relabeling and exploratory pretraining [[Yu et al., 2022]](#ref-yuHowLeverageUnlabeled2022); [[Yarats et al., 2022]](#ref-yaratsDontChangeAlgorithm2022).

<!-- _footer: "Ref: [[Chow & Ghavamzadeh, 2014]](#ref-chowAlgorithmsCVaROptimization2014); [[Yu et al., 2022]](#ref-yuHowLeverageUnlabeled2022)" -->

---

## 17. Summary & Takeaways

- **Offline RL transitions RL from interactive trial-and-error to data-driven learning**, opening control to high-stakes real-world domains.
- **Distributional shift** is the core hurdle: mitigated through policy constraints, value regularization (CQL), uncertainty estimation, or sequence modeling.
- **Algorithmic simplicity often wins**: TD3+BC and IQL demonstrate that avoiding over-complex generative behavior models provides higher reliability.
- **Next Frontiers**: Robust off-policy selection, self-supervised representations on unlabeled data, and safe real-world deployment workflows.

<!-- _footer: "Ref: [[Prudencio et al., 2024]](#ref-prudencioSurveyOfflineReinforcement2024)" -->

---

<!-- _class: lead -->
# References

---

## References (1/5)

<div style="font-size: 14px; line-height: 1.4;">

- <a id="ref-prudencioSurveyOfflineReinforcement2024"></a> **[Prudencio et al., 2024]** Prudencio, R., Maximo, M., & Colombini, E. (2024). *A Survey on Offline Reinforcement Learning: Taxonomy, Review, and Open Problems*. IEEE Transactions on Neural Networks and Learning Systems. https://doi.org/10.1109/TNNLS.2023.3250269
- <a id="ref-precupEligibilityTracesOffpolicy2000"></a> **[Precup et al., 2000]** Precup, D., Sutton, R. S., & Singh, S. P. (2000). *Eligibility Traces for Off-Policy Policy Evaluation*. ICML.
- <a id="ref-chowAlgorithmsCVaROptimization2014"></a> **[Chow & Ghavamzadeh, 2014]** Chow, Y. & Ghavamzadeh, M. (2014). *Algorithms for CVaR Optimization in MDPs*. NeurIPS.
- <a id="ref-jiangDoublyRobustOffpolicy2016"></a> **[Jiang & Li, 2016]** Jiang, N. & Li, L. (2016). *Doubly Robust Off-Policy Value Evaluation for Reinforcement Learning*. ICML.
- <a id="ref-hallakConsistentOnlineOffpolicy2017"></a> **[Hallak & Mannor, 2017]** Hallak, A. & Mannor, S. (2017). *Consistent on-Line off-Policy Evaluation*. ICML.
- <a id="ref-fujimotoAddressingFunctionApproximation2018"></a> **[Fujimoto et al., 2018]** Fujimoto, S., Hoof, H., & Meger, D. (2018). *Addressing Function Approximation Error in Actor-Critic Methods*. ICML.
- <a id="ref-haarnojaSoftActorcriticOffpolicy2018"></a> **[Haarnoja et al., 2018]** Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*. ICML.
- <a id="ref-wangExponentiallyWeightedImitation2018"></a> **[Wang et al., 2018]** Wang, Q., Xiong, J., Han, L., Sun, P., Liu, H., & Zhang, T. (2018). *Exponentially Weighted Imitation Learning for Batched Historical Data*. NeurIPS.

</div>

---

## References (2/5)

<div style="font-size: 14px; line-height: 1.4;">

- <a id="ref-fujimotoOffPolicyDeepReinforcement2018"></a> **[Fujimoto et al., 2018]** Fujimoto, S., Meger, D., & Precup, D. (2018). *Off-Policy Deep Reinforcement Learning without Exploration*. arXiv. https://doi.org/10.48550/arXiv.1812.02900
- <a id="ref-kumarStabilizingOffPolicyQLearning2019"></a> **[Kumar et al., 2019]** Kumar, A., Fu, J., Tucker, G., & Levine, S. (2019). *Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction*. arXiv. https://doi.org/10.48550/arXiv.1906.00949
- <a id="ref-wuBehaviorRegularizedOffline2019"></a> **[Wu et al., 2019]** Wu, Y., Tucker, G., & Nachum, O. (2019). *Behavior Regularized Offline Reinforcement Learning*. arXiv. https://doi.org/10.48550/arXiv.1911.11361
- <a id="ref-pengAdvantageWeightedRegressionSimple2019"></a> **[Peng et al., 2019]** Peng, X., Kumar, A., Zhang, G., & Levine, S. (2019). *Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning*. arXiv. https://doi.org/10.48550/arXiv.1910.00177
- <a id="ref-nachumAlgaeDICEPolicyGradient2019"></a> **[Nachum et al., 2019]** Nachum, O., Dai, B., Kostrikov, I., Chow, Y., Li, L., & Schuurmans, D. (2019). *AlgaeDICE: Policy Gradient from Arbitrary Experience*. arXiv. https://doi.org/10.48550/arXiv.1912.02074
- <a id="ref-voloshinEmpiricalStudyOffpolicy2019"></a> **[Voloshin et al., 2019]** Voloshin, C., Le, H. M., Jiang, N., & Yue, Y. (2019). *Empirical Study of Off-Policy Policy Evaluation for Reinforcement Learning*. arXiv. https://doi.org/10.48550/arXiv.1911.06854
- <a id="ref-fuD4RLDatasetsDeep2020"></a> **[Fu et al., 2020]** Fu, J., Kumar, A., Nachum, O., Tucker, G., & Levine, S. (2020). *D4RL: Datasets for Deep Data-Driven Reinforcement Learning*. arXiv. https://doi.org/10.48550/arXiv.2004.07219
- <a id="ref-kumarConservativeQLearningOffline2020"></a> **[Kumar et al., 2020]** Kumar, A., Zhou, A., Tucker, G., & Levine, S. (2020). *Conservative Q-Learning for Offline Reinforcement Learning*. arXiv. https://doi.org/10.48550/arXiv.2006.04779

</div>

---

## References (3/5)

<div style="font-size: 14px; line-height: 1.4;">

- <a id="ref-yuMOPOModelbasedOffline2020"></a> **[Yu et al., 2020]** Yu, T., Thomas, G., Yu, L., Ermon, S., Zou, J., Levine, S., Finn, C., & Ma, T. (2020). *MOPO: Model-based Offline Policy Optimization*. arXiv. https://doi.org/10.48550/arXiv.2005.13239
- <a id="ref-chenBAILBestActionImitation2020"></a> **[Chen et al., 2020]** Chen, X., Zhou, Z., Wang, Z., Wang, C., Wu, Y., & Ross, K. (2020). *BAIL: Best-Action Imitation Learning for Batch Deep Reinforcement Learning*. arXiv. https://doi.org/10.48550/arXiv.1910.12179
- <a id="ref-siegelKeepDoingWhat2020"></a> **[Siegel et al., 2020]** Siegel, N., Springenberg, J., Berkenkamp, F., Abdolmaleki, A., Neunert, M., Lampe, T., Hafner, R., Heess, N., & Riedmiller, M. (2020). *Keep Doing What Worked: Behavioral Modelling Priors for Offline Reinforcement Learning*. arXiv. https://doi.org/10.48550/arXiv.2002.08396
- <a id="ref-agarwalOptimisticPerspectiveOffline2020"></a> **[Agarwal et al., 2020]** Agarwal, R., Schuurmans, D., & Norouzi, M. (2020). *An Optimistic Perspective on Offline Reinforcement Learning*. arXiv. https://doi.org/10.48550/arXiv.1907.04543
- <a id="ref-matsushimaDeploymentEfficientReinforcementLearning2020"></a> **[Matsushima et al., 2020]** Matsushima, T., Furuta, H., Matsuo, Y., Nachum, O., & Gu, S. (2020). *Deployment-Efficient Reinforcement Learning via Model-Based Offline Optimization*. arXiv. https://doi.org/10.48550/arXiv.2006.03647
- <a id="ref-gulcehreRLUnpluggedSuite2020"></a> **[Gulcehre et al., 2020]** Gulcehre, C., Wang, Z., Novikov, A., Paine, T., Gómez, S., Sygnowski, K., Merel, J., Mehrjou, A., et al. (2020). *RL Unplugged: A Suite of Benchmarks for Offline Reinforcement Learning*. NeurIPS.
- <a id="ref-paineHyperparameterSelectionOffline2020"></a> **[Paine et al., 2020]** Paine, T. L., Paduraru, C., Michi, A., Gulcehre, C., Wang, Z., & de Freitas, N. (2020). *Hyperparameter Selection for Offline Reinforcement Learning*. arXiv. https://doi.org/10.48550/arXiv.2007.09055
- <a id="ref-fuBenchmarksDeepOffpolicy2021"></a> **[Fu et al., 2021]** Fu, J., Norouzi, M., Nachum, O., Tucker, G., Wang, Z., Novikov, A., Meng, M., et al. (2021). *Benchmarks for Deep Off-Policy Evaluation*. ICLR.

</div>

---

## References (4/5)

<div style="font-size: 14px; line-height: 1.4;">

- <a id="ref-kostrikovOfflineReinforcementLearning2021a"></a> **[Kostrikov et al., 2021a]** Kostrikov, I., Tompson, J., Fergus, R., & Nachum, O. (2021). *Offline Reinforcement Learning with Fisher Divergence Critic Regularization*. arXiv. https://doi.org/10.48550/arXiv.2103.08050
- <a id="ref-kostrikovOfflineReinforcementLearning2021"></a> **[Kostrikov et al., 2021b]** Kostrikov, I., Nair, A., & Levine, S. (2021). *Offline Reinforcement Learning with Implicit Q-Learning*. arXiv. https://doi.org/10.48550/arXiv.2110.06169
- <a id="ref-nairAWACAcceleratingOnline2021"></a> **[Nair et al., 2021]** Nair, A., Gupta, A., Dalal, M., & Levine, S. (2021). *AWAC: Accelerating Online Reinforcement Learning with Offline Datasets*. arXiv. https://doi.org/10.48550/arXiv.2006.09359
- <a id="ref-fujimotoMinimalistApproachOffline2021"></a> **[Fujimoto & Gu, 2021]** Fujimoto, S. & Gu, S. (2021). *A Minimalist Approach to Offline Reinforcement Learning*. arXiv. https://doi.org/10.48550/arXiv.2106.06860
- <a id="ref-wangCriticRegularizedRegression2021"></a> **[Wang et al., 2021]** Wang, Z., Novikov, A., Zolna, K., Springenberg, J., Reed, S., Shahriari, B., Siegel, N., Merel, J., Gulcehre, C., Heess, N., & Freitas, N. (2021). *Critic Regularized Regression*. arXiv. https://doi.org/10.48550/arXiv.2006.15134
- <a id="ref-brandfonbrenerOfflineRLOffPolicy2021"></a> **[Brandfonbrener et al., 2021]** Brandfonbrener, D., Whitney, W., Ranganath, R., & Bruna, J. (2021). *Offline RL Without Off-Policy Evaluation*. arXiv. https://doi.org/10.48550/arXiv.2106.08909
- <a id="ref-chenDecisionTransformerReinforcement2021"></a> **[Chen et al., 2021]** Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A., & Mordatch, I. (2021). *Decision Transformer: Reinforcement Learning via Sequence Modeling*. arXiv. https://doi.org/10.48550/arXiv.2106.01345
- <a id="ref-jannerOfflineReinforcementLearning2021"></a> **[Janner et al., 2021]** Janner, M., Li, Q., & Levine, S. (2021). *Offline Reinforcement Learning as One Big Sequence Modeling Problem*. arXiv. https://doi.org/10.48550/arXiv.2106.02039

</div>

---

## References (5/5)

<div style="font-size: 14px; line-height: 1.4;">

- <a id="ref-kidambiMOReLModelBasedOffline2021"></a> **[Kidambi et al., 2021]** Kidambi, R., Rajeswaran, A., Netrapalli, P., & Joachims, T. (2021). *MOReL : Model-Based Offline Reinforcement Learning*. arXiv. https://doi.org/10.48550/arXiv.2005.05951
- <a id="ref-yuCOMBOConservativeOffline2022"></a> **[Yu et al., 2022]** Yu, T., Kumar, A., Rafailov, R., Rajeswaran, A., Levine, S., & Finn, C. (2022). *COMBO: Conservative Offline Model-Based Policy Optimization*. arXiv. https://doi.org/10.48550/arXiv.2102.08363
- <a id="ref-emmonsRvSWhatEssential2022"></a> **[Emmons et al., 2022]** Emmons, S., Eysenbach, B., Kostrikov, I., & Levine, S. (2022). *RvS: What is Essential for Offline RL via Supervised Learning?*. arXiv. https://doi.org/10.48550/arXiv.2112.10751
- <a id="ref-emersonOfflineReinforcementLearning2022"></a> **[Emerson et al., 2022]** Emerson, H., Guy, M., & McConville, R. (2022). *Offline Reinforcement Learning for Safer Blood Glucose Control in People with Type 1 Diabetes*. arXiv. https://doi.org/10.48550/arXiv.2204.03376
- <a id="ref-vermaCHAIChatbotAITaskoriented2022"></a> **[Verma et al., 2022]** Verma, S., Fu, J., Yang, S., & Levine, S. (2022). *CHAI: A Chatbot AI for Task-Oriented Dialogue with Offline Reinforcement Learning*. NAACL.
- <a id="ref-zhanDeepThermalCombustionOptimization2022"></a> **[Zhan et al., 2022]** Zhan, X., Xu, H., Zhang, Y., Zhu, X., Yin, H., & Zheng, Y. (2022). *DeepThermal: Combustion Optimization for Thermal Power Generating Units Using Offline Reinforcement Learning*. AAAI.
- <a id="ref-yuHowLeverageUnlabeled2022"></a> **[Yu et al., 2022]** Yu, T., Kumar, A., Chebotar, Y., Hausman, K., Finn, C., & Levine, S. (2022). *How to Leverage Unlabeled Data in Offline Reinforcement Learning*. ICML.
- <a id="ref-yaratsDontChangeAlgorithm2022"></a> **[Yarats et al., 2022]** Yarats, D., Brandfonbrener, D., Liu, H., Laskin, M., Abbeel, P., Lazaric, A., & Pinto, L. (2022). *Don't Change the Algorithm, Change the Data: Exploratory Data for Offline Reinforcement Learning*. arXiv. https://doi.org/10.48550/arXiv.2201.13425

</div>
