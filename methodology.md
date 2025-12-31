## Methodology (experiments + evaluation)

This document describes the **implemented experiments** in `experiments/` and how each reported metric corresponds to (or approximates) the mathematical quantities defined in `main.tex`.

The code lives in `experiments/src/forecastbench/`. The main entrypoint is:

- `PYTHONPATH=src python -m forecastbench ...` (run from `experiments/`)

Artifacts (configs, metrics, plots, predictions) are written to `experiments/runs/<timestamp>_<run_name>/`.

---

## 1) Common setup and notation (maps to `main.tex`)

### 1.1 Forecasting model

We follow the binary-forecasting setup in `main.tex` (Sections “Forecasting Setup…” and “Proper Scoring Rules…”).

- **Context / information**: \(X \in \mathcal{X}\).
- **Outcome**: \(Y \in \{0,1\}\).
- **Truth function**:

\[
f(x) \;:=\; \mathbb{P}(Y=1 \mid X=x).
\]

- **Forecaster**: a function \(q:\mathcal{X}\to(0,1)\) (the mean prediction of a possibly-randomized model \(M\) in `main.tex`).

In code:
- synthetic benchmarks treat \(X\) as a Boolean vector \(Z\in\{-1,+1\}^d\) (or a text encoding of it);
- Polymarket benchmarks treat \(X\) as a text prompt built from dataset columns (e.g. question/description).

### 1.2 Sample vs. population quantities

`main.tex` states most results at the **population** level (expectations over the data-generating distribution). The harness computes **sample estimates** on finite datasets:

- Synthetic: i.i.d. samples \((Z_i,Y_i)\).
- Polymarket: a dataset of resolved events \((x_i,y_i)\) with optional market price features.

Unless noted, every “\(\mathbb{E}[\cdot]\)” below is estimated by an empirical mean over the evaluation split.

---

## 2) Metrics (exact definitions used by the harness)

### 2.1 Proper losses: Brier and log (maps to `main.tex` \(\ell_{\mathrm{Br}},\ell_{\log}\))

For predictions \(q_i\in(0,1)\) and labels \(y_i\in\{0,1\}\):

- **Brier loss** (quadratic / Brier):

\[
\mathrm{Brier}(q,y)\;=\;\frac{1}{n}\sum_{i=1}^n (q_i-y_i)^2.
\]

This matches `main.tex` definition \(\ell_{\mathrm{Br}}(q,y)=(q-y)^2\) and is implemented in `experiments/src/forecastbench/metrics/proper.py` (`brier_loss`).

- **Log loss** (Bernoulli negative log-likelihood / Kelly loss):

\[
\mathrm{LogLoss}(q,y)\;=\;\frac{1}{n}\sum_{i=1}^n -\Big(y_i\log \tilde q_i + (1-y_i)\log(1-\tilde q_i)\Big),
\quad \tilde q_i := \mathrm{clip}(q_i,\varepsilon,1-\varepsilon),
\]

with \(\varepsilon=10^{-6}\) in code (`log_loss`).

**Relation to `main.tex` regret quantities.**
In `main.tex` Proposition “Proper scoring regret decomposition” (and “Brier-to-Kelly conversion under clipping”), the *population regret* under Brier is exactly \(\mathbb{E}[(q(X)-f(X))^2]\) (SCE), and the log-loss regret is a Bernoulli KL divergence. In empirical evaluation on real data we typically report absolute log loss and compare models by differences in log loss (regret differences).

### 2.2 Squared calibration error (SCE) (maps to `main.tex` \(\mathrm{SCE}\))

On **synthetic** benchmarks the Truth function \(f\) is known, so we can compute:

\[
\mathrm{SCE}(q;f)\;:=\;\mathbb{E}\big[(q(X)-f(X))^2\big]
\approx \frac{1}{n}\sum_{i=1}^n (q_i - f_i)^2.
\]

This is exactly `experiments/src/forecastbench/metrics/proper.py::squared_calibration_error`.

On **real** datasets (e.g. Polymarket), \(f(x)\) is unknown, so SCE is not directly observable; we instead rely on proper losses (Brier/log) + calibration diagnostics (ECE, group biases).

### 2.3 Expected calibration error (ECE) (binning diagnostic; compares to `main.tex` robust calibration views)

The harness computes a standard equal-width-bin ECE:

- Partition \([0,1]\) into \(m\) bins \(B_1,\dots,B_m\).
- For each bin \(B_b\), let \(\bar q_b\) be the average prediction in the bin and \(\bar y_b\) be the average label.
- ECE is:

\[
\mathrm{ECE}(q,y) \;=\; \sum_{b=1}^m \frac{n_b}{n}\,|\bar y_b-\bar q_b|.
\]

Implemented in `experiments/src/forecastbench/metrics/calibration.py::expected_calibration_error`.

**How this compares to `main.tex`.**
`main.tex` defines robust calibration as a supremum over test functions (and group calibration as a supremum over groups). ECE is a commonly-used *binned* approximation that is not identical to \(\mathrm{Cal}_\mathcal{H}\) or \(\mathrm{GCal}_\mathcal{G}\), but it is directionally useful and cheap to estimate.

### 2.4 Bounded statistical arbitrage proxy (synthetic; maps to `main.tex` \(\mathrm{ArbReg}\) and \(\mathbb{E}|f-q|\))

For synthetic benchmarks with known truth probabilities \(p_i=f(x_i)\), the harness reports:

\[
\mathrm{ArbProfit}(B,c)
\;:=\;
B \cdot \mathbb{E}\big[(|p-q|-c)_+\big]
\approx
B\cdot \frac{1}{n}\sum_{i=1}^n (|p_i-q_i|-c)_+.
\]

This is implemented as `experiments/src/forecastbench/metrics/arbitrage.py::best_bounded_trader_profit`.

**Mapping to `main.tex`.**
This quantity matches the per-round optimum of a bounded static trader under linear payoff with optional transaction cost (see `main.tex` Proposition “Best bounded statistical arbitrage equals \(L^1\) mispricing” and the cost-adjusted variant).

### 2.5 Group calibration bias (real-data; maps to `main.tex` group calibration \(\mathrm{GCal}\))

For a dataset column `group_col` defining a finite partition into groups \(g\), we compute:

- **Worst-group absolute bias**:

\[
\max_{g}\; \left|\mathbb{E}[\,Y-q(X)\mid g\,]\right|.
\]

- **Average absolute bias** (group-size-weighted):

\[
\sum_g \mathbb{P}(g)\,\left|\mathbb{E}[\,Y-q(X)\mid g\,]\right|.
\]

Implemented in `experiments/src/forecastbench/benchmarks/polymarket_eval.py::group_calibration_bias`.

This is a concrete instantiation of `main.tex` group calibration error \(\mathrm{GCal}_{\mathcal{G}}\) when \(\mathcal{G}\) is the set of groups induced by the chosen column(s).

### 2.6 Realized trading PnL proxy (real-data; evaluation-only)

If a dataset contains a traded price column `market_prob` (interpreted as the market price at the time you would trade), we compute a simple realized PnL proxy:

- Choose a position \(b_i\in[-B,B]\) from the discrepancy between our forecast \(p_i\) and market price \(q_i\).
- Realize:

\[
\mathrm{PnL}_i \;=\; b_i\,(y_i-q_i)\;-\;c\,|b_i|.
\]

The harness supports two rules:
- `sign`: \(b_i = B\cdot \mathrm{sign}(p_i-q_i)\)
- `linear`: \(b_i = \mathrm{clip}(B(p_i-q_i),-B,B)\)

Reported as average realized PnL per event (`pnl_per_event`) in `experiments/src/forecastbench/benchmarks/polymarket_eval.py::realized_trading_pnl`.

**Important caveat.**
This is not the in-hindsight optimum \(\sup_{b(\cdot)}\mathbb{E}[b(Y-q)]\) from `main.tex`; it is an evaluation proxy that uses *our* forecast to decide a trade against the market quote.

---

## 3) Synthetic experiments (directly aligned with the theory)

All synthetic experiments are implemented in `experiments/src/forecastbench/cli.py` and are designed to match the “Parity markets”, “Small-group stress tests”, and “Intrinsic robustness vs post-processing” protocol in `main.tex` (Section “Experimental Protocol…”).

### 3.1 `parity`: Parity markets + “cliff vs fog” metrics

**Command**: `forecastbench parity`

**Data-generating process.**
Fix \(d\), choose a hidden subset \(S\subseteq[d]\) with \(|S|=k\), and sample \(Z\sim\mathrm{Unif}(\{-1,+1\}^d)\). Define the parity character:

\[
\chi_S(z)\;:=\;\prod_{i\in S} z_i.
\]

Truth probability (matches `main.tex` Equation “parity”):

\[
f(z)\;=\;\frac12 + \frac{\alpha}{2}\chi_S(z),
\qquad \alpha\in(0,1].
\]

Then sample \(Y\mid Z=z \sim \mathrm{Bern}(f(z))\).

**Models compared.**
The benchmark evaluates multiple forecasters \(q(z)\):

- **Oracle**: \(q(z)=f(z)\).
- **Constant**: \(q(z)\equiv 1/2\) (the best “uninformative” baseline, and the correct optimum if a model cannot correlate with \(\chi_S\)).
- **Analytic diffusion**: \(q_{\mathrm{diff},\rho}(z) = (T_\rho f)(z)\). For parity this has the closed form (matches `main.tex` diffusion discussion):

\[
q_{\mathrm{diff},\rho}(z)\;=\;\frac12+\frac{\alpha}{2}\rho^k\chi_S(z).
\]

- **\(L\)-query oracle surrogate**: a stylized AR-like control that succeeds iff \(L\ge k\), else defaults to \(1/2\). This is a sanity check against the `main.tex` “AR cutoff” abstraction (the true LLM is optional).
- **Optional HF LLM (AR+CoT)**: if `--llm-model` is provided, the harness queries a HuggingFace causal LM with a prompt that requests a probability; `L` controls the allowed “step budget” in the prompt and `K` controls self-consistency width. Aggregation is `mean` or `median` to match the evaluation style used in `arXiv:2505.17989`.

**Reported metrics.**
On the evaluation sample the harness reports Brier, log loss, ECE, SCE, and bounded-trader profit \(B\mathbb{E}|f-q|\) (with optional transaction cost). The key paper-aligned metric is **SCE**.

### 3.2 `groupstress`: Subcube small-group diagnostic for parity

**Command**: `forecastbench groupstress`

This is the empirical version of the `main.tex` parity small-group argument (Section “Group Robustness…”).

**Groups.**
For a subset \(J\subseteq[d]\) with \(|J|=k\) and assignment \(a\in\{-1,+1\}^J\), define a subcube group:

\[
G_{J,a}\;:=\;\{z\in\{-1,+1\}^d:\ z_J=a\}.
\]

**Residual and group bias.**
For a fixed forecaster \(q\), define the residual \(r(z)=f(z)-q(z)\). The group calibration bias on \(G_{J,a}\) is:

\[
\mu(J,a)\;:=\;\mathbb{E}[r(Z)\mid Z\in G_{J,a}].
\]

The benchmark computes the worst-case absolute conditional mean for each \(J\):

\[
\max_{a\in\{-1,+1\}^k} |\mu(J,a)|.
\]

**Parity-specific prediction.**
For parity and diffusion \(q=T_\rho f\), `main.tex` shows that the only \(J\) with nonzero conditional expectation is \(J=S\) (when \(|J|=|S|=k\)). The benchmark:

- enumerates many/all \(J\) of size \(k\) and reports the top-\(J\) conditional biases;
- always computes \(J=S\) explicitly; and
- reports the theoretical value \(\frac{\alpha}{2}(1-\rho^k)\) (matching `main.tex` Proposition “Diffusion group robustness on parity markets” specialized to \(J=S\)).

### 3.3 `intrinsic_post`: Intrinsic robustness vs. post-processing (synthetic control)

**Command**: `forecastbench intrinsic_post`

This is the synthetic control suggested in `main.tex` Section “Intrinsic robustness vs post-processing”.

**Base forecasters.**
It compares two base families:

- **AR-intrinsic**: either an HF LLM (`--llm-model`) or the stylized \(L\)-query surrogate (succeeds iff \(L\ge k\), else \(1/2\)).
- **Diffusion-intrinsic**: analytic \(q_{\mathrm{diff},\rho}=T_\rho f\) for parity.

**Post-processing wrapper (shared across AR and diffusion).**
We fit the same simple group-conditional binning calibrator on the training split:

- groups are the \(2^k\) subcubes induced by the *true* parity coordinates \(S\);
- within each group we bin predictions into `post_bins` bins, and estimate \(\mathbb{E}[Y\mid \text{group},\text{bin}]\) with a Beta-like smoothing prior of strength `post_prior`.

This is implemented in `experiments/src/forecastbench/postprocess.py`.

**What this isolates.**
Because the family has \(2^k\) rare groups, post-processing exhibits the **group-frequency/sample tax** discussed in `main.tex` (finite-sample limits for small-group evaluation): learning a separate correction for each rare group requires enough samples in each group.

**Reported metrics.**
In addition to the parity metrics, `intrinsic_post` reports:

- `gcal_S`: \(\max_a |\mathbb{E}[f(Z)-q(Z)\mid Z_S=a]|\), i.e. worst-group conditional bias on the parity-induced partition (a direct proxy for worst-group calibration in the parity family).

---

## 4) Simplex (multi-outcome) synthetic experiment (diffusion-to-simplex infrastructure)

### 4.1 ALR transform (maps “text → simplex” requirement into \(\mathbb{R}^{m-1}\))

For an \(m\)-outcome categorical market with probabilities \(p\in\Delta^{m-1}\), we use the additive log-ratio (ALR) map:

\[
\mathrm{alr}(p)_i \;=\; \log\frac{p_i}{p_m},\quad i=1,\dots,m-1,
\qquad
\mathrm{alr}^{-1}(u)_i \;=\; \frac{e^{u_i}}{1+\sum_{j=1}^{m-1}e^{u_j}},\quad
\mathrm{alr}^{-1}(u)_m \;=\; \frac{1}{1+\sum_{j=1}^{m-1}e^{u_j}}.
\]

Implemented in `experiments/src/forecastbench/utils/logits.py` (`simplex_to_alr`, `alr_to_simplex`).

### 4.2 `difftrain_simplex`: Learned simplex diffusion sanity check (training; not evaluation-only)

**Command**: `forecastbench difftrain_simplex`

This trains a tiny conditional diffusion model in ALR space on a synthetic “multi-outcome parity” task:

- define \(u(z)\in\mathbb{R}^{m-1}\) with coordinates \(u_i(z)=\alpha\chi_{S_i}(z)\),
- set \(p(z)=\mathrm{alr}^{-1}(u(z))\),
- sample \(Y\sim\mathrm{Categorical}(p(z))\).

Reported metrics are the multi-class analogues:

- Multi-class Brier: \(\mathbb{E}\|q-e_Y\|_2^2\)
- Multi-class log loss: \(\mathbb{E}[-\log q_Y]\)
- Multi-class SCE: \(\mathbb{E}\|q-p\|_2^2\)

Implemented in `experiments/src/forecastbench/metrics/multiclass.py`.

---

## 5) Polymarket evaluation-only experiments (aligned to `arXiv:2505.17989` intent)

The guiding goal is the “evaluation parity” request: we focus on **evaluation** (proper scoring + calibration + a trading proxy) rather than RLVR training loops.

### 5.1 Dataset schema (minimal)

All Polymarket-style datasets must include at least:

- `id`: stable identifier
- `question`: the prompt
- `y`: resolved label in \(\{0,1\}\)

Validated by `experiments/src/forecastbench/data/dataset.py`.

Optional columns used by parts of the pipeline include:

- `description`, `category`, timestamps (`createdAt`, `closedTime`, …)
- `market_prob`: a market price you can treat as the traded quote for a trading simulation

### 5.2 Data acquisition / preprocessing commands (may require network)

These commands build the offline Parquet dataset that `pm_eval` consumes:

- `pm_download_gamma`: download market metadata via Polymarket’s public Gamma API into JSONL (+ progress tracking).
- `pm_build_gamma`: build a clean yes/no Parquet dataset from a Gamma dump (filters, labels, token IDs).
- `pm_enrich_clob`: (optional) fill `market_prob` using the Polymarket CLOB historical price API (a step toward “forecast-time” prices; requires network access).
- `pm_build_polydata`, `pm_build_subgraph`: alternative ingestion routes (manual JSON export or GraphQL query).

### 5.3 `pm_eval`: Evaluation-only on a Polymarket-style dataset

**Command**: `forecastbench pm_eval`

**Inputs.**
Choose:
- a dataset `--dataset-path`;
- a prediction column `--pred-col` containing \(q(x)\in[0,1]\),
  or provide `--llm-model` to have the harness produce predictions into `pred_col` from text columns.

Text input is built by concatenating the chosen `--text-cols` (e.g. `question,description`).

**Metrics.**
The harness reports:

- proper scoring: Brier, log loss;
- calibration diagnostic: ECE;
- optional group calibration bias for discrete columns `--group-cols` (worst/avg absolute bias);
- optional realized trading PnL proxy if `market_prob` exists (trade against the market quote using `sign` or `linear` rule).

Implemented in `experiments/src/forecastbench/benchmarks/polymarket_eval.py`.

**Comparison to `arXiv:2505.17989`.**
This mirrors the evaluation-only component: “given a question/context, output a probability; evaluate with proper scoring and calibration; simulate a simple trading rule against market odds when available”.

### 5.4 Commands that *train* models (not run in evaluation-only mode)

These exist but should be treated as optional baselines/sanity checks rather than the main evaluation protocol:

- `difftrain`: trains a tiny conditional logit-diffusion model on parity (synthetic sanity).
- `pm_difftrain`: trains a conditional logit-diffusion model on a Polymarket dataset using HF text embeddings, then evaluates on a held-out split.

---

## 6) What “evaluation-only” means operationally

When we say **evaluation-only**, we mean:

- run synthetic benchmarks that do not fit neural networks (`parity`, `groupstress`, `intrinsic_post` with the lightweight post-processor),
- run `pm_eval` using an existing prediction column (e.g. a market baseline), without running `pm_difftrain` or any finetuning.

If you want to include an HF LLM for AR+CoT evaluation, that is still “evaluation-only” (inference), but it requires model weights and potentially GPU resources.




