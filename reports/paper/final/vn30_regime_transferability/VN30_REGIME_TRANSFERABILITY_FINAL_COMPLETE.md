# Regime-Dependent Transferability of Machine Learning Forecasts in an Emerging Equity Market: Evidence from VN30 Walk-Forward Benchmarks

## Abstract

This paper examines whether machine-learning forecasts for VN30 constituent stocks transfer across reconstructed latent market regimes. The study uses a walk-forward benchmark as a discovery and diagnostic engine, not as evidence of a trading system or as proof that the reconstructed regimes are structural economic states. The primary latent-regime evidence is evaluated at the h40 horizon, using a train-only 3-regime GaussianMixture fit on 259 training timestamps and lagged state features covering return, volatility, breadth, dispersion, and volume-shock context. Model, feature, and threshold selection are governed by validation evidence; final-window outputs are retained only as descriptive scoring context. The bounded h40 benchmark result is Logistic L2 with the C-closest reference feature set, validation-selected threshold 0.55, final accuracy 61.63%, and full 30-stock coverage. Regime diagnostics support a cautious interpretation. H1 is diagnostically supported because validation Regime Information Gain is positive, with RIG = 0.403333. H2 is diagnostically supported because validation forecast correctness differs across reconstructed regimes, with an accuracy gap of 0.085666, chi-square = 121.98, and p = 3.25e-27. H3 is mixed and partially supported: cross-regime transfer behavior is non-identical, with TRR ranging from 0.826009 to 1.169407 and TG ranging from -0.074978 to 0.090714, but transfer loss is not uniform. H4 receives only narrow partial support: the original K=3 validation distance tests are nonsignificant, and the strengthening audit finds 3 supported rows only for K=4 train-only GMM, ticker-level observations, RD_cosine, balanced-accuracy transfer metrics, and log-loss gaps. The resulting contribution is a bounded diagnostic audit of regime-conditioned transfer behavior rather than a claim about economic regime causality or deployable forecasting performance. RD and FRD remain diagnostic distances, not causal mechanisms.

## 1. Introduction

Machine-learning studies in equity forecasting often face a transfer problem. A model can perform well in one validation period, one market state, or one forecast horizon, yet weaken when evaluated under a later state or different conditional environment. This problem is particularly relevant for emerging equity markets, where liquidity, turnover, transaction costs, and investor behavior can affect return dynamics and prediction stability (Rouwenhorst, 1999; Chang, Cheng, and Khorana, 2000; Lesmond, 2005; Bekaert, Harvey, and Lundblad, 2007).

This paper studies that problem through VN30 stock-level directional forecasting. The question is not whether a single model can produce the highest final-window score. The question is whether forecast relationships and forecast transferability vary across reconstructed latent regimes, and whether distance between those regimes is associated with weaker transfer.

The benchmark is used as a diagnostic engine. It helps identify and test candidate regime-transferability patterns, but it does not prove that the reconstructed regimes are structural economic states. The primary empirical evidence for the latent-regime claims comes from validation. Final-window outputs are used only as descriptive scoring context and cannot replace the validation-governed h40 result.

The central contribution is a claim-bounded regime-transferability audit for VN30. The paper reports four hypothesis tests. H1 asks whether regime information adds predictive value. H2 asks whether forecasting skill differs across regimes. H3 asks whether forecast skill is imperfectly transferable across regimes. H4 asks whether transferability decreases as regime distance increases. The results support H1 and H2 diagnostically, support H3 only in a mixed and partial sense, and support H4 only under a narrow K=4 ticker-level robustness design rather than as a general monotonic distance-transfer law.

The headline h40 benchmark result remains deliberately conservative: Logistic L2 using the C-closest reference feature set, validation-selected threshold 0.55, final accuracy 61.63%, and full coverage of the 30 VN30 constituents. This result is a historical diagnostic finding, not a live trading system, profitability result, investment recommendation, or general claim outside the reported VN30 benchmark.

The paper proceeds as follows. Section 2 reviews efficient-market, adaptive-market, time-varying predictability, machine-learning, validation, latent-state, probability-evaluation, and distribution-shift literature. Section 3 defines the conceptual framework and diagnostic metrics. Section 4 states the hypotheses. Section 5 describes the data, walk-forward design, regime construction, transfer tests, and H4 strengthening audit. Section 6 reports the empirical results. Sections 7 through 9 discuss interpretation, limitations, and conclusions.

![Figure 1. Research Design Flow. The empirical design moves from a VN30 walk-forward benchmark through latent-regime tests and ends with an explicit claim boundary.](figures/figure1_research_design_flow.png)

**Table 1. Study Design and Claim Boundary**

| Element | Claim-safe statement | Boundary |
| --- | --- | --- |
| Benchmark role | Discovery and diagnostic engine only | Does not prove economic regimes or general market inefficiency. |
| Main h40 claim | The bounded h40 benchmark result is 61.63% final accuracy under validation-only selection and full VN30 stock coverage. | Validation-only selection; full 30-stock coverage. |
| Router 63.33% | The 63.33% router score is reported only as descriptive final-window context. | Descriptive final-window context only. |
| Soft voting near 62% | Soft-voting results near 62% are context for cooperation diagnostics, not claim-eligible evidence. | Cooperation diagnostic only. |
| Forbidden scope | No trading, profitability, investment, production-deployment, unreported score-target, or broad generalization claim. | Claim boundary preserved. |

## 2. Literature Review

Financial forecasting research starts from a demanding benchmark: if markets are efficient, persistent return prediction from public market information should be difficult. Fama (1970) frames efficient capital markets as settings in which prices reflect available information, and weak-form efficiency implies that historical price and trading information should not provide stable exploitable predictability. This benchmark does not rule out finite-sample diagnostic patterns, but it does require caution before interpreting prediction accuracy as durable market inefficiency.

Adaptive-market theory provides the main motivation for treating predictability as conditional rather than fixed. Lo (2004) argues that market efficiency can evolve with market ecology, competition, learning, and changing institutional conditions. In that view, predictability is not expected to be constant across all periods. Instead, the same model may perform differently across states that reflect volatility, breadth, dispersion, volume shock, trend, liquidity, or participation conditions.

Empirical AMH evidence supports this time-varying perspective without proving that any specific VN30 result must hold. Kim, Shamsuddin, and Lim (2011) document time-varying return predictability in century-long U.S. data. Lim and Brooks (2011) survey evidence on the evolution of stock-market efficiency over time. Urquhart and Hudson (2013) provide long-run evidence from major U.S., U.K., and Japanese equity markets that is consistent with an adaptive-market interpretation. These studies motivate the paper's focus on changing predictive relationships, while leaving the VN30 evidence to be established by the paper's own validation design.

Evidence from Asian and emerging-market settings further motivates a conditional approach. Xiong et al. (2019) examine calendar effects in China through an AMH lens, showing that strategy effects can vary over time. The Vietnam AMH study by Dzung Phan Tran Trung and Hung Pham Quang (2019) provides Vietnamese market-context motivation using HSX and HNX index evidence. That Vietnam evidence is used here only as market-context motivation; it is not a substitute for this paper's VN30 stock-level regime-transferability tests.

Machine-learning methods have also been widely applied to equity forecasting tasks. Fischer and Krauss (2018) provide general evidence that LSTM models can be used for S&P 500 constituent forecasting, which motivates machine-learning relevance in equity prediction. This paper uses that source only as general ML forecasting context. It does not use LSTM evidence from another market to claim that the VN30 benchmark is profitable, live deployable, or externally generalizable.

Time-series validation literature emphasizes that forecast evaluation should respect temporal dependence and out-of-sample ordering, while financial machine-learning practice adds leakage controls for model selection and backtesting (Bergmeir and Benitez, 2012; Lopez de Prado, 2018). However, temporal splitting alone does not solve all selection problems. If model families, thresholds, regime routers, or headline claims are chosen after inspecting final scores, the final window stops being an independent transfer assessment. This paper therefore distinguishes validation-governed evidence from descriptive final-window scoring.

Regime-aware financial modeling also motivates this study. Hamilton (1989) provides a classic Markov-switching framework for latent regime changes in economic time series, and Ang and Bekaert (2002) apply regime-shift modeling to international asset allocation. Bull, bear, sideway, high-volatility, low-liquidity, and drawdown states may change feature-outcome relationships. In this paper, the regimes are not manually labeled economic states. They are reconstructed latent states from lagged market and stock-context features. This distinction matters because the paper tests whether those reconstructed states carry diagnostic information; it does not claim that they are definitive economic regimes.

Probability forecast evaluation is also relevant when models produce scores or probabilities rather than only hard labels. The Brier score originates in probability-forecast verification (Brier, 1950), proper scoring rules provide a general framework for evaluating probabilistic forecasts (Gneiting and Raftery, 2007), and supervised-learning calibration research shows that predicted probabilities may require separate calibration assessment (Niculescu-Mizil and Caruana, 2005).

Finally, transferability and distribution shift remain important issues in financial machine learning. General dataset-shift literature defines the problem of differing train and test distributions (Quinonero-Candela et al., 2008), while financial prediction studies document instability and weak out-of-sample behavior in return-prediction models (Paye and Timmermann, 2006; Welch and Goyal, 2008). A model trained under one conditional state may not retain its skill in another state. This paper operationalizes that issue with same-regime and cross-regime transfer cells, transfer-retention ratios, transfer gaps, and diagnostic distance measures. The diagnostic distances used here are not interpreted as causal mechanisms, and the literature reviewed above does not provide general support for H4.

**Table 7. References and Literature Role**

| Literature role | Verified citation(s) | Use in this paper |
| --- | --- | --- |
| EMH / weak-form efficiency | Fama (1970) | Benchmark for difficulty of persistent prediction. |
| AMH | Lo (2004) | Motivates time-varying efficiency and conditional predictability. |
| Time-varying predictability | Kim, Shamsuddin, and Lim (2011); Lim and Brooks (2011) | Supports evolving return predictability. |
| International AMH evidence | Urquhart and Hudson (2013) | Motivates but does not prove VN30 findings. |
| China and Vietnam context | Xiong et al. (2019); Dzung Phan Tran Trung and Hung Pham Quang (2019) | Context only; not substitute VN30 evidence. |
| ML equity forecasting | Fischer and Krauss (2018) | General ML forecasting context only. |
| Validation discipline | Bergmeir and Benitez (2012); Lopez de Prado (2018) | Supports temporal and leakage-aware evaluation. |
| Latent-state finance | Hamilton (1989); Ang and Bekaert (2002) | Precedent for state-dependent financial dynamics. |
| Probability evaluation | Brier (1950); Gneiting and Raftery (2007); Niculescu-Mizil and Caruana (2005) | Supports scoring and calibration evaluation. |
| Dataset shift / instability | Quinonero-Candela et al. (2008); Paye and Timmermann (2006); Welch and Goyal (2008) | Motivates transferability caution. |
| Emerging-market context | Rouwenhorst (1999); Chang, Cheng, and Khorana (2000); Lesmond (2005); Bekaert, Harvey, and Lundblad (2007) | Motivates liquidity, turnover, cost, and behavior context. |

## 3. Conceptual Framework

### 3.1 Market Efficiency, Adaptation, and Conditional Predictability

Efficient-market logic implies that unconditional, persistent directional predictability should be difficult. Adaptive-market logic allows predictability to vary with market ecology, liquidity, participation, and state transitions. The two views are not treated as mutually exclusive here. Instead, they motivate a conditional question: if predictive relationships exist in this VN30 benchmark, are they stable across reconstructed regimes?

The paper does not argue that regime conditioning defeats market efficiency. It asks whether latent state information improves diagnostic forecast evidence in validation and whether forecast relationships transfer across state boundaries.

### 3.2 Regime-Conditional Forecasting

Regime-conditional forecasting assumes that the mapping from features to future direction may differ by state. Let \(R_t\) denote the reconstructed latent regime at time \(t\), \(X_t\) the feature vector, and \(Y_{t+h}\) the directional target at horizon \(h\). A regime-conditional forecast allows:

\[
P(Y_{t+h}=1 \mid X_t, R_t=i) \neq P(Y_{t+h}=1 \mid X_t, R_t=j)
\]

for at least some regimes \(i\) and \(j\). The empirical question is whether conditioning on \(R_t\) improves validation diagnostics or reveals forecast-correctness differences across reconstructed regimes.

### 3.3 Forecast Relationship Shift

Forecast relationship shift means that the learned relation between predictors and future direction differs across regimes. In this paper, relationship shift is assessed through diagnostic evidence such as same-regime log-loss relative to global log-loss, cross-regime transfer behavior, and forecast relationship distance. These diagnostics can be consistent with shifting relationships, but they are not a formal structural-break proof.

### 3.4 Transferability Loss

Transferability loss occurs when a model or relationship learned in one regime performs worse when evaluated in another regime. For regimes \(i\) and \(j\), the audit compares same-regime performance for \(i \rightarrow i\) with cross-regime performance for \(i \rightarrow j\). Loss is not assumed to be universal. A cross-regime cell can improve, weaken, or behave similarly depending on the specific pair.

### 3.5 Regime Distance and Forecast Relationship Distance

Regime Distance (RD) measures standardized latent-regime centroid distance over lagged state columns. Forecast Relationship Distance (FRD) measures distance between regime-specific forecast relationships using coefficient L2 distance, coefficient cosine distance, and predicted probability distribution distance. Where probabilities are available, the audit also examines Brier score gaps and calibration error gaps.

RD and FRD are diagnostic distances. They are not causal mechanisms. A positive association between distance and transfer loss would support H4 only if it meets the pre-specified statistical criteria. It would not establish that distance causes transfer loss.

### 3.6 Diagnostic Metrics

The paper uses four primary diagnostic metrics:

\[
RIG = LogLoss_{global} - LogLoss_{regime}
\]

\[
TRR_{ij} = \frac{Accuracy_{cross}(i \rightarrow j)}{Accuracy_{same}(i \rightarrow i)}
\]

\[
TG_{ij} = Accuracy_{same}(i \rightarrow i) - Accuracy_{cross}(i \rightarrow j)
\]

\[
RD_{ij} = standardized\ latent-regime\ centroid\ distance
\]

Positive RIG indicates lower validation log-loss when regime information is used. TRR below 1 indicates weaker cross-regime transfer than same-regime performance. Positive TG indicates an accuracy loss when moving from same-regime to cross-regime evaluation. Larger RD indicates greater standardized separation between latent-regime centroids.

## 4. Hypotheses

H1: Regime information adds predictive value. This hypothesis is supported if validation regime conditioning reduces log-loss relative to the global model. Final status: diagnostically supported.

H2: Forecasting skill differs across regimes. This hypothesis is supported if forecast correctness differs across reconstructed latent regimes in validation. Final status: diagnostically supported.

H3: Forecasting skill is not perfectly transferable across regimes. This hypothesis is supported if cross-regime transfer cells differ from same-regime reference behavior. Because transfer losses are mixed in sign and magnitude, the final status is mixed and partially supported.

H4: Forecasting transferability decreases as regime distance increases. This hypothesis is supported only when distance-transfer tests show the expected direction, p < 0.05, and consistency across at least two related metrics or two horizons. The final status is partial support under a narrow robustness design and not general support.

## 5. Data and Methodology

### 5.1 VN30 Walk-Forward Benchmark

The empirical setting is VN30 stock-level directional prediction. The benchmark covers full 30-stock headline coverage and evaluates multiple forecast horizons, including h20, h40, h60, and h80. The main bounded result is reported at h40.

The design separates training, validation, and final scoring. Training data are used to fit models, preprocessing, and latent-regime construction. Validation data govern model selection, threshold selection, and hypothesis testing. Final-window scores are descriptive scoring-only outputs.

The headline h40 benchmark row is Logistic L2 using the C-closest reference feature set, with threshold 0.55 selected on validation evidence. Its final-window accuracy is 61.63% with full 30-stock coverage.

### 5.2 Latent-Regime Construction

The primary latent-regime analysis uses a train-only 3-regime GaussianMixture for h40. The fit uses 259 train unique timestamps. The state features are lagged and cover return, volatility, breadth, dispersion, and volume-shock context.

The regime labels are reconstructed from local lagged features. They are not external bull, bear, sideway, crisis, or macroeconomic labels. The feature audit reports no future regime labels, no future return features, no same-row target leakage, and no final-window-derived features.

**Table 2. Latent-Regime Construction Audit**

| Audit item | Value | Interpretation |
| --- | --- | --- |
| Method | train-only 3-regime GaussianMixture | Fit on training timestamps only. |
| Horizon | h40 | Primary latent-regime diagnostics in this pack. |
| Training timestamps | 259 | Unique training dates used for GMM fit. |
| State features | Lagged return, volatility, breadth, dispersion, and volume-shock context. | No external economic labels. |
| Lagged variables | yes | State columns are ex-ante or lagged. |
| Leakage audit | no target leakage flags in feature-family audit | No target/future/final leakage flags reported. |
| Final outputs | yes | Scoring-only; not used for selection. |

Initial paper-pack robustness did not run K=2/K=4. A later H4 strengthening audit computed K=2 and K=4 variants; K=4 produced the only supported H4 rows, while K=2 mostly produced inconclusive or degenerate-distance tests.

### 5.3 Regime-Transfer Tests

The transfer analysis compares same-regime and cross-regime cells. For each source-target regime pair, the audit computes transfer-retention and transfer-gap diagnostics. Same-regime cells provide the \(i \rightarrow i\) reference. Cross-regime cells provide the \(i \rightarrow j\) transfer evidence.

The analysis emphasizes validation because validation is the primary empirical window for the latent-regime evidence. Final-window latent-regime outputs are descriptive and are not used to promote stronger claims.

### 5.4 H4 Strengthening Audit

The H4 strengthening audit tests whether distance-linked transfer decay receives stronger legitimate support under pre-specified additional designs. It uses existing local artifacts and local feature builders only.

The audit adds FRD variants: coefficient L2 distance, coefficient cosine distance, predicted probability distribution distance, Brier score gap where probabilities are available, and calibration error gap where feasible. It also evaluates multi-horizon diagnostics for h20, h40, h60, and h80 when labels and features are available without a heavy rerun.

For regime-count robustness, the audit computes train-only K=2 and K=4 GMM assignments when lightweight fitting is feasible. For each K, it recomputes RD variants, transfer cells, TRR, TG, and log-loss gap tests. The K=4 design is the only design that produces supported H4 rows, and the support is limited to ticker-level RD_cosine against balanced-accuracy TRR/TG and logloss_gap. K=2 mostly produces inconclusive or degenerate-distance tests.

The statistical criteria are conservative. A test is supported only when it has the expected direction, p < 0.05, and consistency across at least two related metrics or two horizons. Expected-direction but nonsignificant rows, or isolated significant rows in one narrow metric, are weak. Wrong-direction or near-zero relationships are not supported. Insufficient or unstable estimates are inconclusive.

## 6. Results

![Figure 2. Hypothesis Evidence Status Panel. H1 and H2 are diagnostically supported, H3 is mixed/partially supported, and H4 is only narrowly partially supported.](figures/figure2_hypothesis_status_panel.png)

**Table 3. Hypothesis Evidence Summary**

| Hypothesis | Key evidence | Final status | Claim-safe wording |
| --- | --- | --- | --- |
| H1 | validation RIG = 0.403333 | diagnostically supported | H1 is diagnostically supported in validation by positive RIG; this is diagnostic evidence, not proof of economic regime existence. |
| H2 | accuracy gap = 0.085666; chi2 = 121.98; p = 3.25e-27 | diagnostically supported | H2 is diagnostically supported: validation forecast correctness differs across reconstructed latent regimes. |
| H3 | TRR range = 0.826009 to 1.169407; TG range = -0.074978 to 0.090714 | mixed / partially supported | H3 is mixed/partially supported: transfer behavior differs by regime pair, but cross-regime loss is not uniform. |
| H4 | Original K=3 validation RD tests nonsignificant; strengthening audit: 3 supported, 294 weak, 557 not supported, 196 inconclusive. | partially supported under narrow robustness design; not general support | H4 is partially supported only under a narrow robustness design using K=4 train-only GMM, ticker-level observations, cosine-based regime distance, and balanced-accuracy/log-loss transfer metrics. It is not established as a general distance-transfer law. |

### 6.1 Hypothesis Evidence

Table 3 summarizes the hypothesis-level evidence. H1 and H2 are diagnostically supported in validation. H3 is mixed and partially supported because transfer behavior differs across regime pairs but does not show uniform cross-regime deterioration. H4 receives partial support only under a narrow K=4 ticker-level robustness design and is not established as a general distance-transfer law.

### 6.2 H1: Regime Information Gain

H1 is diagnostically supported in validation. The Regime Information Gain is 0.403333, indicating that same-regime validation log-loss is lower than global validation log-loss under the diagnostic construction. This supports the interpretation that the reconstructed latent regime state is information-bearing for validation diagnostics.

This result does not prove the existence of economic regimes. It shows that the reconstructed state variable is associated with improved validation log-loss in this benchmark.

### 6.3 H2: Regime-Conditional Forecasting Skill

H2 is diagnostically supported. Forecast correctness differs across reconstructed latent regimes in validation. The validation accuracy gap is 0.085666, with chi-square = 121.98 and p = 3.25e-27. This is strong diagnostic evidence that forecast correctness is not uniform across latent regimes.

The result is interpreted as regime-conditional diagnostic evidence. It is not a full conditional-distribution proof and does not establish a live trading edge.

**Table 4. Regime Information and Regime-Conditional Accuracy**

| Diagnostic | Value | Role | Interpretation |
| --- | --- | --- | --- |
| RIG (validation) | 0.403333 | primary | Primary diagnostic evidence: same-regime log-loss is lower than global log-loss, giving positive validation RIG. |
| RIG (final) | 0.429032 | descriptive_only | Descriptive final-window scoring only; not selection evidence and not promoted beyond context. |
| Accuracy association (validation) | gap 0.085666; chi-square 121.98; p 3.25e-27 | primary | Primary diagnostic evidence: global-model correctness differs across latent regimes in validation. |
| Accuracy association (final) | gap 0.118453; chi-square 44.28; p 2.4232e-10 | descriptive_only | Descriptive final-window scoring only; not selection evidence. |

### 6.4 H3: Transferability Across Regimes

H3 is mixed and partially supported. Cross-regime transfer cells are not identical. TRR ranges from 0.826009 to 1.169407, and TG ranges from -0.074978 to 0.090714. These ranges show both deterioration and improvement across different regime pairs.

The correct interpretation is regime-pair heterogeneity. Some cross-regime transfers lose accuracy relative to same-regime references, but transfer loss is not uniform. The paper therefore avoids a claim that all cross-regime transfers deteriorate.

**Table 5. Transferability Summary**

| Split | TRR range | TG range | TG signs | Interpretation |
| --- | --- | --- | --- | --- |
| validation | 0.826009 to 1.169407 | -0.074978 to 0.090714 | +3 / -3 | Cross-regime TG has mixed signs, so H3 is mixed/partially supported rather than uniformly supported. |
| final | 0.682257 to 1.263797 | -0.138643 to 0.212575 | +4 / -2 | Descriptive final-window context only; cross-regime signs are not selection evidence. |

![Figure 3. Transfer Matrix Heatmap. Validation transfer-retention ratios show mixed cross-regime behavior and are diagnostic only.](figures/figure3_transfer_matrix_heatmap.png)

### 6.5 H4: Distance-Linked Transferability

H4 receives only bounded partial support. The original K=3 validation RD-vs-TRR and RD-vs-TG tests are nonsignificant. The strengthening audit finds 3 supported rows, 294 weak rows, 557 not supported rows, and 196 inconclusive rows.

The supported rows appear only under a narrow robustness design: K=4 train-only GMM, ticker-level observations, RD_cosine, TRR_balanced_accuracy, TG_balanced_accuracy, and logloss_gap. The supported K=4 rows have expected directions and p < 0.05 across related metric families. K=2 mostly produces inconclusive or degenerate-distance tests.

FRD and multi-horizon tests do not reach supported status. FRD variants report 0 supported, 15 weak, 27 not supported, and 14 inconclusive tests. Multi-horizon tests report 0 supported, 256 weak, 458 not supported, and 56 inconclusive tests.

The claim-safe conclusion is that H4 has partial support only in a narrow K=4 ticker-level robustness design. The broader evidence does not establish a general monotonic distance-transfer mechanism. RD and FRD remain diagnostic, not causal.

**Table 6. H4 Strengthening Audit Summary**

| Scope | Supported | Weak | Not supported | Inconclusive | Claim-safe wording |
| --- | --- | --- | --- | --- | --- |
| overall | 3 | 294 | 557 | 196 | H4 is partially supported under specific robustness designs, not established as a general law. |
| regime_count | 3 | 23 | 72 | 126 | H4 is partially supported under specific robustness designs, not established as a general law. |
| frd | 0 | 15 | 27 | 14 | H4 is partially supported under specific robustness designs, not established as a general law. |
| multihorizon | 0 | 256 | 458 | 56 | H4 is partially supported under specific robustness designs, not established as a general law. |

*Note.* Supported rows appear only for K=4 train-only GMM, ticker-level observations, RD_cosine, TRR_balanced_accuracy, TG_balanced_accuracy, and logloss_gap. FRD and multi-horizon tests do not reach supported status.

![Figure 4. H4 Robustness Count Panel. The strengthening audit has 3 supported rows, while most tests are weak, not supported, or inconclusive.](figures/figure4_h4_robustness_count_panel.png)

### 6.6 Main h40 Benchmark Result and Descriptive Final-Window Context

The main bounded h40 benchmark result remains Logistic L2 using the C-closest reference feature set, validation-selected threshold 0.55, final accuracy 61.63%, and full 30-stock coverage. This result remains the claim-eligible h40 benchmark row because it follows the validation-governed selection boundary.

The bull/bear/sideway router reaches 63.33% final-window accuracy at h40, but it is not claim-eligible because it is descriptive final-window context. Soft-voting accuracy near 62.00% is also descriptive context only. These rows can motivate further research but do not replace the main bounded h40 result.

Market-index evidence is treated as context only. It does not substitute for stock-level VN30 evidence.

## 7. Discussion

The results show that regime information matters diagnostically in this VN30 benchmark. H1 and H2 provide validation evidence that reconstructed latent regimes contain information relevant to forecast log-loss and forecast correctness. This supports the use of latent regimes as diagnostic state variables.

The transfer results are more cautious. H3 indicates that transfer behavior differs across regime pairs, but the direction is mixed. Some pairs show lower cross-regime retention and positive transfer gaps, while others show equal or improved transfer. This is consistent with regime-pair heterogeneity rather than universal deterioration.

The H4 result is the most limited. The original K=3 validation tests do not support a distance-linked transfer decay claim. The strengthening audit improves the status only within a narrow K=4 ticker-level design using RD_cosine and balanced-accuracy or log-loss transfer metrics. Because FRD and multi-horizon tests fail to reach supported status, the evidence should be described as a bounded robustness finding, not as a general distance-transfer law.

### 7.1 Theoretical Interpretation of Partial Support

The pattern of evidence should be interpreted as partial theoretical support rather than as rejection of the proposed framework. H1 and H2 provide the strongest support because they are both grounded in validation evidence. H1 indicates that reconstructed latent regimes contain additional predictive diagnostic information: same-regime validation log-loss is lower than the global reference, producing positive Regime Information Gain. This does not mean that the reconstructed states are proven economic regimes. It means that, within the reported VN30 benchmark, the locally reconstructed state variable is information-bearing for forecast diagnostics.

H2 strengthens this interpretation by showing that forecast correctness differs across reconstructed regimes in validation. This supports the Regime-Conditional Forecasting Proposition: predictive performance is not fully invariant across market-state contexts. The result is still diagnostic rather than structural. It does not prove that the conditional data-generating process has been fully identified, but it does show that the same global forecasting relationship behaves differently across the reconstructed states used in this paper.

H3 adds a more nuanced implication. Mixed and partial support means that transferability loss is heterogeneous across regime pairs, not absent. Some cross-regime cells show weaker transfer relative to same-regime references, while others behave similarly or improve. The theoretical implication is that transferability cannot be reduced to a simple all-or-nothing claim. Forecast relationships appear partly state-dependent, but the direction and magnitude of transfer loss depend on the source-target regime pair and the metric used to summarize transfer behavior.

H4 is the narrowest result and should be read as a bounded robustness finding. Market State Distance may explain part of transferability loss under specific conditions, especially in the K=4 train-only GMM, ticker-level robustness design using RD_cosine and balanced-accuracy or log-loss transfer metrics. However, it is not a complete, general, or causal mechanism. The original K=3 validation tests are nonsignificant, FRD variants do not reach supported status, and multi-horizon tests do not reach supported status. The distance-based hypothesis therefore remains theoretically plausible but empirically incomplete.

Taken together, H3 and H4 suggest that forecasting transferability is more complex than the initial distance mechanism. A richer account may need to separate several channels that are only indirectly represented by the current latent-state features. Future extensions could examine liquidity shocks, investor sentiment, macroeconomic conditions, algorithm-specific sensitivity, richer distance measures, and future-blind validation windows. These extensions would test whether the observed transfer heterogeneity reflects broader market-state dynamics or narrower sample-specific interactions.

This interpretation also explains why partial support is scientifically informative. A fully uniform transfer-loss pattern would imply a simple regime-separation story, while the observed mixed pattern suggests that some market states may share usable predictive structure despite being distinct under the reconstruction. The framework therefore points toward conditional transfer maps rather than a single universal transfer penalty.

The overall theoretical contribution remains bounded but meaningful. The evidence indicates that forecasting relationships are not fully invariant across reconstructed market regimes and that forecasting skill is partly conditional on market state. The paper therefore supports a regime-aware diagnostic perspective while preserving the claim boundary: latent regimes remain reconstructed diagnostic states, RD and FRD remain diagnostic distances, final-window outputs remain descriptive scoring context, and the benchmark does not imply trading, profitability, production deployment, or broad generalization.

This interpretation also clarifies the role of the broader benchmark. The benchmark can identify candidate patterns and motivate formal diagnostic testing. It does not, by itself, prove regime existence, structural breaks, or tradable economic mechanisms. The final-window router and soft-voting rows are useful diagnostics but remain outside the claim-eligible headline result.

![Figure 5. RD/FRD Claim Boundary Diagram. Diagnostic distances are not causal mechanisms, and descriptive final-window outputs do not create trading, profitability, or live-deployment claims.](figures/figure5_rd_frd_claim_boundary.png)

## 8. Limitations and Future Work

Several limitations follow directly from the design. First, the regimes are reconstructed from local lagged features rather than externally validated economic labels. They should therefore be interpreted as latent diagnostic states.

Second, final-window latent-regime outputs are descriptive scoring-only evidence. They should not be used to strengthen claims that are not already governed by validation evidence.

Third, H4 support is narrow. K=4 ticker-level RD_cosine evidence supports H4 under the audit criteria, but K=3 validation tests, FRD variants, and multi-horizon tests do not provide broad support. Future work should pre-specify distance metrics, regime-count choices, and transfer metrics before applying them to new future-blind windows.

Fourth, the paper does not test a trading strategy. It does not include transaction costs, liquidity constraints, turnover, risk limits, slippage, portfolio construction, or live execution. Accuracy diagnostics should not be interpreted as profitability.

Future work should evaluate whether the K=4 ticker-level finding survives in later windows, alternative markets, externally defined market states, and unchanged data and model rules. Future work should also examine whether forecast relationship distance can be stabilized through richer calibration diagnostics, larger samples, or independent validation cohorts.

## 9. Conclusion

This paper provides a bounded VN30 regime-transferability audit. H1 and H2 are diagnostically supported in validation: regime information improves log-loss diagnostics, and forecast correctness differs across reconstructed latent regimes. H3 is mixed and partially supported: transfer behavior differs across regime pairs, but transfer loss is not uniform. H4 receives partial support only under a narrow K=4 ticker-level robustness design and remains not general support.

The main claim-eligible h40 benchmark result remains Logistic L2 using the C-closest reference feature set, validation-selected threshold 0.55, final accuracy 61.63%, and full 30-stock coverage. Higher descriptive final-window rows, including the 63.33% router and soft-voting near 62.00%, do not replace that result.

The paper's contribution is therefore diagnostic and bounded. It shows that reconstructed latent regimes are useful for understanding validation performance and transfer heterogeneity in the VN30 benchmark, while avoiding claims about trading profitability, production deployment, causal distance mechanisms, unreported score-target performance, or generalization beyond the reported historical setting.

## References

Ang, Andrew, and Geert Bekaert. 2002. "International Asset Allocation with Regime Shifts." Review of Financial Studies 15(4): 1137-1187. DOI: 10.1093/rfs/15.4.1137.

Bekaert, Geert, Campbell R. Harvey, and Christian Lundblad. 2007. "Liquidity and Expected Returns: Lessons from Emerging Markets." Review of Financial Studies 20(6): 1783-1831. DOI: 10.1093/rfs/hhm030.

Bergmeir, Christoph, and Jose M. Benitez. 2012. "On the use of cross-validation for time series predictor evaluation." Information Sciences 191: 192-213. DOI: 10.1016/j.ins.2011.12.028.

Brier, Glenn W. 1950. "Verification of Forecasts Expressed in Terms of Probability." Monthly Weather Review 78(1): 1-3. DOI: 10.1175/1520-0493(1950)078\<0001:VOFEIT\>2.0.CO;2.

Chang, Eric C., Joseph W. Cheng, and Ajay Khorana. 2000. "An Examination of Herd Behavior in Equity Markets: An International Perspective." Journal of Banking & Finance 24(10): 1651-1679. DOI: 10.1016/S0378-4266(99)00096-5.

Dzung Phan Tran Trung, and Hung Pham Quang. 2019. "Adaptive Market Hypothesis: Evidence from the Vietnamese Stock Market." Journal of Risk and Financial Management 12(2): 81. DOI: 10.3390/jrfm12020081.

Fama, Eugene F. 1970. "Efficient Capital Markets: A Review of Theory and Empirical Work." Journal of Finance 25(2): 383-417.

Fischer, Thomas, and Christopher Krauss. 2018. "Deep learning with long short-term memory networks for financial market predictions." European Journal of Operational Research 270(2): 654-669. DOI: 10.1016/j.ejor.2017.11.054.

Gneiting, Tilmann, and Adrian E. Raftery. 2007. "Strictly Proper Scoring Rules, Prediction, and Estimation." Journal of the American Statistical Association 102(477): 359-378. DOI: 10.1198/016214506000001437.

Hamilton, James D. 1989. "A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle." Econometrica 57(2): 357-384. DOI: 10.2307/1912559.

Kim, Jae H., Abul Shamsuddin, and Kian-Ping Lim. 2011. "Stock return predictability and the adaptive markets hypothesis: Evidence from century-long U.S. data." Journal of Empirical Finance 18(5): 868-879. DOI: 10.1016/j.jempfin.2011.08.002.

Lesmond, David A. 2005. "Liquidity of emerging markets." Journal of Financial Economics 77(2): 411-452. DOI: 10.1016/j.jfineco.2004.01.005.

Lim, Kian-Ping, and Robert Brooks. 2011. "The evolution of stock market efficiency over time: A survey of the empirical literature." Journal of Economic Surveys 25(1): 69-108. DOI: 10.1111/j.1467-6419.2009.00611.x.

Lo, Andrew W. 2004. "The Adaptive Markets Hypothesis: Market Efficiency from an Evolutionary Perspective." Journal of Portfolio Management 30(5): 15-29. DOI: 10.3905/jpm.2004.442611.

Lopez de Prado, Marcos. 2018. Advances in Financial Machine Learning. Hoboken, NJ: Wiley. ISBN: 9781119482086.

Niculescu-Mizil, Alexandru, and Rich Caruana. 2005. "Predicting good probabilities with supervised learning." Proceedings of the 22nd International Conference on Machine Learning: 625-632. DOI: 10.1145/1102351.1102430.

Paye, Bradley S., and Allan Timmermann. 2006. "Instability of return prediction models." Journal of Empirical Finance 13(3): 274-315. DOI: 10.1016/j.jempfin.2005.11.001.

Quinonero-Candela, Joaquin, Masashi Sugiyama, Anton Schwaighofer, and Neil D. Lawrence, eds. 2008. Dataset Shift in Machine Learning. Cambridge, MA: MIT Press. ISBN: 9780262170055.

Rouwenhorst, K. Geert. 1999. "Local Return Factors and Turnover in Emerging Stock Markets." Journal of Finance 54(4): 1439-1464. DOI: 10.1111/0022-1082.00151.

Urquhart, Andrew, and Robert Hudson. 2013. "Efficient or adaptive markets? Evidence from major stock markets using very long run historic data." International Review of Financial Analysis 28: 130-142. DOI: 10.1016/j.irfa.2013.03.005.

Welch, Ivo, and Amit Goyal. 2008. "A Comprehensive Look at The Empirical Performance of Equity Premium Prediction." Review of Financial Studies 21(4): 1455-1508. DOI: 10.1093/rfs/hhm014.

Xiong, Xiong, Yongqiang Meng, Xiao Li, and Dehua Shen. 2019. "An empirical analysis of the Adaptive Market Hypothesis with calendar effects: Evidence from China." Finance Research Letters 31(C). DOI: 10.1016/j.frl.2018.11.020.
