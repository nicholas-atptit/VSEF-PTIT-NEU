# Regime-Dependent Transferability of Machine Learning Forecasts in an Emerging Equity Market: Evidence from VN30 Walk-Forward Benchmarks

<style>
@page {
  size: A4;
  margin: 22mm 21mm 22mm 21mm;
}
body {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 12pt;
  line-height: 1.38;
}
h1, h2, h3 {
  line-height: 1.2;
}
table {
  font-size: 10pt;
  line-height: 1.25;
}
img {
  max-width: 100%;
}
</style>

## Abstract

This paper examines whether machine-learning forecasts for VN30 constituent stocks transfer across reconstructed latent market regimes, using the benchmark as a discovery and diagnostic design rather than as evidence of a trading system, causal market mechanism, or deployable forecasting process. The primary claim-eligible analysis is conducted at h40 with a train-only 3-regime GaussianMixture fitted on 259 train unique timestamps and lagged state features that summarize return, volatility, breadth, dispersion, and volume-shock context. Model, feature, and threshold choices are governed by validation evidence. The bounded h40 benchmark result remains Logistic L2 using the C-closest reference feature set, validation-selected threshold 0.55, final accuracy 61.63%, and full 30-stock coverage. Final-window outputs are retained only as descriptive scoring context.

The validation evidence supports a regime-aware diagnostic interpretation. H1 is diagnostically supported because regime conditioning improves log-loss relative to the global reference, with validation RIG = 0.403333. H2 is diagnostically supported because forecast correctness differs across reconstructed regimes, with an accuracy gap = 0.085666, chi-square = 121.98, and p = 3.25e-27. H3 is mixed and partially supported because transfer behavior is non-identical across regime pairs, with validation TRR = 0.826009 to 1.169407 and TG = -0.074978 to 0.090714, but cross-regime deterioration is not uniform. H4 remains narrowly bounded: the strengthening audit reports H4 = 3 supported, 294 weak, 557 not supported, 196 inconclusive, and support appears only under a K=4 ticker-level RD_cosine design, using balanced-accuracy transfer metrics and log-loss gaps.

The paper contributes a claim-bounded empirical framework for evaluating regime-dependent forecast transferability in an emerging equity market. It extends adaptive-market reasoning into machine-learning forecast diagnostics by separating regime information, regime-conditional skill, cross-regime transfer behavior, and distance-linked transfer hypotheses. The evidence does not establish a causal distance mechanism, broad VN30 generalization, trading profitability, investment relevance, production deployment, or live-deployment readiness. RD and FRD are interpreted only as diagnostic distances. Router 63.33% and soft voting near 62.00% remain descriptive final-window context and are not claim-eligible. The resulting contribution is a bounded diagnostic audit showing that reconstructed latent regimes help explain validation performance and transfer heterogeneity without overstating what the benchmark can prove.

## 1. Introduction

Machine-learning research in equity forecasting is often organized around rankings. A study defines a universe of models, constructs train and test splits, reports accuracy or loss, and identifies the best-performing row. That convention is useful for benchmarking, but it is incomplete when the scientific question concerns stability. A model can rank first in one window and still be fragile across market states, forecast horizons, ticker groups, or conditional environments. In financial data, the central concern is not only whether a model can score above a baseline in a particular historical sample. It is also whether the relationship learned by the model remains informative when the market context changes.

This paper studies that problem in VN30 stock-level directional forecasting. The motivating issue is transferability: whether forecasting relationships learned or evaluated under one reconstructed latent regime remain useful under another. The paper therefore treats the benchmark as a diagnostic engine. The benchmark is not used to prove a trading edge, to certify a production system, or to assert that the reconstructed regimes are structural economic states. It is used to organize a disciplined empirical audit of regime information, regime-conditional skill, cross-regime transfer behavior, and distance-linked transfer diagnostics.

Emerging equity markets make the transferability problem especially important. Liquidity, trading costs, institutional participation, turnover, market access, and investor behavior can differ from those observed in highly developed markets (Rouwenhorst, 1999; Chang, Cheng, and Khorana, 2000; Lesmond, 2005; Bekaert, Harvey, and Lundblad, 2007). These features can make predictive relationships unstable over time or uneven across market states. In such settings, a final-window model ranking alone can be misleading because it compresses heterogeneous conditions into a single number. A single headline accuracy can hide whether performance is concentrated in one regime, whether some regimes are systematically harder, and whether a model selected under one state transfers poorly into another.

The VN30 setting is useful for this audit because it allows the analysis to stay at the stock level while retaining a clearly defined constituent universe. The headline h40 result is deliberately conservative: Logistic L2 using the C-closest reference feature set, validation-selected threshold 0.55, final accuracy 61.63%, and full coverage of the 30 VN30 constituents. That row is claim-eligible only because selection is governed by validation evidence and the coverage boundary is explicit. Other final-window observations, including router 63.33% and soft voting near 62.00%, are descriptive context only and do not replace the bounded h40 result.

The research gap is methodological and empirical. Existing model-ranking work can show that one algorithm outperforms another within a chosen split, but it often says less about whether the apparent forecast relationship is state-dependent. Efficient-market reasoning warns that persistent return prediction from public information should be difficult (Fama, 1970), while adaptive-market reasoning allows predictability to vary with market ecology and changing conditions (Lo, 2004). The gap addressed here is the connection between those ideas and machine-learning forecast transferability. The paper asks whether reconstructed latent regimes provide diagnostic information, whether forecast correctness differs across those regimes, whether forecast relationships transfer symmetrically, and whether diagnostic distances are associated with weaker transfer.

The objectives are fourfold. First, the paper evaluates whether regime information improves validation diagnostics. Second, it tests whether forecast correctness differs across reconstructed latent regimes. Third, it measures whether same-regime and cross-regime transfer cells behave similarly or differently. Fourth, it audits whether distance between regimes or forecast relationships is associated with transfer loss under conservative claim criteria. These objectives correspond to four hypotheses: H1 on Regime Information Gain, H2 on regime-conditional forecasting skill, H3 on imperfect cross-regime transferability, and H4 on distance-linked transferability.

The research questions are stated in diagnostic rather than causal form. Does a train-only latent-regime reconstruction carry validation information beyond a global reference? Does model correctness vary across reconstructed regimes? Are cross-regime transfer cells equivalent to same-regime reference cells? Do diagnostic distances such as RD and FRD align with transfer loss strongly enough to support a bounded distance-transfer claim? These questions are intentionally narrower than questions about economic regime causality, trading profitability, or broad generalization.

The paper makes three contributions. The first contribution is a claim-bounded VN30 stock-level regime-transferability audit. The second contribution is a decomposition of regime-aware forecasting evidence into information gain, correctness association, transfer matrices, and distance-linked tests. The third contribution is interpretive: it shows that H1 and H2 can be supported while H3 and H4 remain partial, and that such a pattern is scientifically useful. Partial support means that regime-aware forecasting is informative, but the transfer mechanism is more complex than a simple monotonic distance rule.

The analysis also clarifies what the evidence does not show. Benchmark evidence is discovery and diagnostic only. RD and FRD are diagnostic distances, not causal mechanisms. Final-window outputs are descriptive scoring-only. Vietnam AMH evidence is contextual motivation and does not replace the paper's VN30 stock-level evidence. International AMH evidence motivates time-varying predictability and does not prove VN30 H4. No trading, profitability, investment, production-deployment, live-deployment, unreported score-target, or broad generalization claim is made.

The paper proceeds as follows. Section 2 reviews efficient-market, adaptive-market, emerging-market, validation, latent-state, probability-evaluation, and dataset-shift literature. Section 3 develops the conceptual framework and defines the diagnostic metrics. Section 4 states the hypotheses and links them to empirical tests. Section 5 describes the walk-forward design, leakage controls, train-only GMM construction, transfer-matrix construction, and H4 strengthening audit. Section 6 reports the empirical results. Section 7 discusses theoretical interpretation. Section 8 presents limitations and future work. Section 9 concludes. Appendices state the claim boundary, metric definitions, hypothesis-to-evidence map, H4 interpretation, and table and figure notes.

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

### 2.1 Efficient Markets and Weak-Form Efficiency

Financial forecasting research begins from the efficient-market benchmark. Fama (1970) frames efficient capital markets as markets in which prices reflect available information. In weak-form efficiency, historical prices and trading information should not provide stable, exploitable predictability after appropriate risk and cost considerations. This benchmark is central because many machine-learning equity studies use lagged price, return, volume, and technical information. If those inputs repeatedly generate predictive power without strict validation, the result demands scrutiny.

Weak-form efficiency does not imply that every finite sample must be unpredictable. It does imply that predictive claims should be bounded. A sample can contain temporary regularities, delayed adjustment, liquidity effects, or conditional patterns without proving a durable inefficiency. It also means that a high model rank in a benchmark is not sufficient evidence of a stable forecasting relationship. The stronger the forecasting claim, the more important it becomes to control selection, leakage, and final-window interpretation.

This paper uses efficient-market logic as a discipline rather than as a null that mechanically rejects all diagnostic patterns. The VN30 benchmark can reveal validation relationships, but those relationships are not interpreted as permanent return predictability. H1 and H2 are therefore framed as diagnostic support. They ask whether reconstructed regimes add validation information and whether correctness differs across regimes, not whether public information guarantees persistent profit.

### 2.2 Adaptive Markets and Time-Varying Predictability

Adaptive-market theory provides the main conceptual reason to look beyond unconditional model rankings. Lo (2004) argues that market efficiency evolves with ecology, competition, learning, and institutional conditions. In this view, predictability can appear, disappear, or change form as participants adapt. A model that is useful in one environment may weaken in another because the underlying market ecology has changed.

AMH is especially relevant to regime-transferability analysis because it implies that predictive relationships can be conditional. If volatility, dispersion, breadth, liquidity, or participation changes, the same feature may have a different relationship with future returns. The question is therefore not whether the market is efficient or inefficient in a single unconditional sense. The question is whether the forecasting relationship is stable across reconstructed states.

International AMH evidence motivates this view without proving the VN30 findings. Kim, Shamsuddin, and Lim (2011) document time-varying return predictability in long U.S. data. Lim and Brooks (2011) survey evidence that market efficiency evolves over time. Urquhart and Hudson (2013) provide long-run evidence from major U.S., U.K., and Japanese markets consistent with adaptive-market interpretation. These studies justify a conditional research design, but they do not establish that a particular VN30 latent-regime result must hold.

This distinction is important for H4. AMH can motivate the idea that distance between market states may matter, but it does not prove that a specific RD or FRD metric should monotonically explain transfer loss. The paper therefore treats H4 as an empirical diagnostic question. H4 receives only narrow partial support because the evidence is concentrated in a K=4 ticker-level robustness design and does not generalize across FRD or multi-horizon tests.

### 2.3 Emerging-Market and Vietnam Context

Emerging markets can intensify the need for state-dependent evaluation. Rouwenhorst (1999) documents local return factors and turnover patterns in emerging stock markets. Chang, Cheng, and Khorana (2000) examine herd behavior across equity markets. Lesmond (2005) emphasizes liquidity conditions in emerging markets, and Bekaert, Harvey, and Lundblad (2007) examine liquidity and expected returns in emerging-market settings. These strands motivate caution because observed predictability may interact with liquidity, trading frictions, participation, and market structure.

Vietnam market evidence provides contextual motivation, not a substitute for the paper's own VN30 stock-level tests. Dzung Phan Tran Trung and Hung Pham Quang (2019) examine the Adaptive Market Hypothesis in the Vietnamese stock market using HSX and HNX index evidence. That study helps justify why Vietnam is a relevant setting for time-varying market-efficiency questions. It does not prove that VN30 constituent-level machine-learning forecasts are transferable or non-transferable across reconstructed regimes.

The paper therefore avoids a common overextension. It does not use Vietnam AMH evidence to claim that the VN30 benchmark confirms a market-wide AMH result. Instead, Vietnam evidence is treated as contextual motivation. The empirical burden remains with the paper's own validation design, the h40 benchmark boundary, the train-only 3-regime GaussianMixture, and the hypothesis-specific diagnostics.

### 2.4 Machine Learning in Equity Forecasting

Machine-learning models are attractive in equity forecasting because they can represent nonlinear interactions, high-dimensional feature sets, and changing patterns. Fischer and Krauss (2018) provide evidence that LSTM models can be applied to S&P 500 constituent prediction. That source is relevant because it shows why machine learning is a plausible forecasting technology in equity settings. However, evidence from another market, model family, or sample does not authorize a VN30 claim about profitability.

The stronger lesson for this paper is methodological. Machine-learning benchmarks can produce many candidate rows, thresholds, feature sets, horizons, and ensemble variations. Without a strict claim boundary, the final window can become a selection surface rather than an evaluation surface. A row that looks attractive after repeated inspection may not represent a durable relationship. This problem is not specific to deep learning or to any one algorithm. It applies to any workflow in which researcher choices are influenced by final-window scores.

The h40 headline result is therefore not chosen because it is the highest descriptive score in all available outputs. It is chosen because it remains inside the validation-governed boundary: Logistic L2, C-closest reference feature set, validation-selected threshold 0.55, final accuracy 61.63%, and full 30-stock coverage. Router 63.33% and soft voting near 62.00% are informative diagnostics, but they remain descriptive and are not promoted into claim-eligible results.

### 2.5 Validation, Leakage, and Final-Window Discipline

Time-series validation requires respecting temporal order. Bergmeir and Benitez (2012) discuss cross-validation for time-series predictor evaluation, and Lopez de Prado (2018) emphasizes financial machine-learning controls that prevent leakage and selection bias. In financial forecasting, leakage can arise through future information in features, target-aligned transformations, same-row target leakage, post hoc threshold selection, final-window model choice, or repeated final-window exploration.

Temporal splitting alone is not enough. A study can use chronological train, validation, and final windows while still making claims from a final window that has been indirectly used for selection. The paper therefore separates three roles. Training is used for fitting models, preprocessing, and latent-regime construction. Validation governs model, feature, threshold, and hypothesis evidence. Final-window outputs are descriptive scoring-only.

This discipline affects every main claim. H1 is based on validation Regime Information Gain. H2 is based on validation correctness association. H3 is based on validation transfer ranges, with final transfer ranges described only as context. H4 is bounded by the statistical support pattern and is not strengthened by final-window observations. The final-window RIG = 0.429032 and final descriptive accuracy gap = 0.118453, chi-square = 44.28, p = 2.4232e-10, remain descriptive only.

### 2.6 Latent Regimes and State-Dependent Finance

Regime-aware finance has a long tradition. Hamilton (1989) provides a Markov-switching framework for latent regime changes in economic time series. Ang and Bekaert (2002) apply regime-shift modeling to international asset allocation. These studies motivate the idea that financial dynamics may differ across states. They also illustrate a key distinction: a latent state is an inferred object, not automatically a structural economic label.

In this paper, regimes are reconstructed with a train-only 3-regime GaussianMixture using lagged state features. They are not manually labeled as bull, bear, sideway, crisis, liquidity, or macroeconomic regimes. This protects the claim boundary. A reconstructed state can be useful for forecast diagnostics even if it is not a definitive economic regime. The evidence can show that regime labels are information-bearing without proving why the regimes exist.

This distinction matters for interpretation. H1 and H2 support the diagnostic relevance of the state variable. H3 shows that transfer behavior differs across source-target regime pairs. H4 tests whether distances between reconstructed states explain transfer loss. None of these steps proves economic regime causality.

### 2.7 Probability Evaluation and Calibration

Many machine-learning forecasts produce probabilities or scores before they produce hard labels. Proper evaluation therefore requires more than accuracy. Brier (1950) introduced probability forecast verification through the Brier score. Gneiting and Raftery (2007) provide a general framework for proper scoring rules. Niculescu-Mizil and Caruana (2005) show that predicted probabilities can require separate calibration assessment.

This literature is relevant because RIG and log-loss evaluate probabilistic quality, while transfer metrics such as TRR and TG evaluate hard-label behavior or thresholded decisions under a defined boundary. A model can have useful ranking or classification behavior while being poorly calibrated, and a model can improve log-loss without improving every transfer cell. The paper therefore keeps metric interpretations separate. Positive RIG supports improved regime-conditioned log-loss. It does not automatically imply uniform transferability or calibrated probability behavior across all regimes.

### 2.8 Dataset Shift, Model Instability, and Transferability

Dataset-shift literature describes settings in which training and testing distributions differ (Quinonero-Candela et al., 2008). Financial forecasting is especially exposed to this issue because market conditions evolve, relationships decay, and model performance can be unstable. Paye and Timmermann (2006) document instability in return-prediction models, while Welch and Goyal (2008) provide a broad assessment of weak out-of-sample equity premium prediction performance.

These findings motivate the transferability framing. A model trained or selected under one conditional environment may not retain the same behavior in another. Standard out-of-sample accuracy asks whether a model works on average in a future window. Regime-transferability analysis asks a more granular question: where does it work, where does it weaken, and whether state differences help explain that pattern.

This paper operationalizes transferability through same-regime and cross-regime cells, transfer-retention ratios, transfer gaps, RD, RD_cosine, and FRD variants. The evidence is intentionally conservative. H3 receives mixed and partial support because cross-regime behavior is heterogeneous. H4 receives only narrow partial support because the distance-linked relationship is not stable across all distance designs and horizons.

### 2.9 Literature Positioning

The reviewed literature supports the need for a regime-aware diagnostic audit, but it does not supply the empirical conclusion. EMH provides the cautionary benchmark. AMH motivates conditional predictability. Emerging-market literature explains why Vietnam and VN30 may be sensitive to state-dependent dynamics. Machine-learning and validation literature motivate strict selection discipline. Latent-state and dataset-shift literature justify examining reconstructed regimes and transfer instability.

The paper's position is therefore bounded. It does not claim to settle market efficiency in Vietnam. It does not claim that machine learning produces tradable VN30 predictability. It does not claim that RD or FRD proves a mechanism. It contributes a disciplined empirical map of when validation evidence supports regime-aware forecasting and where transferability claims remain incomplete.

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

The conceptual framework begins from a tension between efficient-market discipline and adaptive-market motivation. Efficient-market logic makes persistent historical-price predictability difficult to claim. Adaptive-market logic suggests that predictability may be episodic, conditional, and shaped by changing market ecology. A regime-transferability framework does not require choosing one view and rejecting the other. It asks whether the empirical behavior of machine-learning forecasts is invariant across reconstructed states.

In this paper, conditional predictability is not interpreted as a direct violation of market efficiency. It is interpreted as a diagnostic property of a bounded historical benchmark. A validation improvement may arise because the latent-state variable organizes the sample into contexts where forecast errors differ. That is informative for model understanding, but it does not establish a permanent forecasting edge or structural market inefficiency.

### 3.2 Regime-Conditional Forecasting

Regime-conditional forecasting assumes that the mapping from observed features to future direction may differ by reconstructed market state. Let \(R_t\) denote the reconstructed latent regime at time \(t\), \(X_t\) the feature vector, \(Y_{t+h}\) the directional target at horizon \(h\), and \(p_{i,t+h}\) the regime-conditional probability of a positive direction.

$$
p_{i,t+h}
=
\Pr(Y_{t+h}=1 \mid X_t, R_t=i)
$$

The regime-conditional forecasting proposition allows at least one pair of regimes to have different conditional forecast relationships:

$$
\exists i \neq j:
\Pr(Y_{t+h}=1 \mid X_t, R_t=i)
\neq
\Pr(Y_{t+h}=1 \mid X_t, R_t=j)
$$

The observed forecast score can then be represented as a function of the model, features, reconstructed regime, and available historical data:

$$
FS_{t+h}
=
g(M, X_t, R_t, \mathcal{D}_t)
$$

These equations are conceptual definitions rather than structural claims. They state what it means for a forecast relationship to be regime-conditional. The empirical sections test whether validation evidence is consistent with this proposition through RIG and correctness association.

### 3.3 Forecast Relationship Shift

Forecast relationship shift occurs when the relation between predictors and the future direction differs across reconstructed regimes. In an equity model, a volatility feature may be informative in one state and less informative in another. A breadth feature may signal persistent participation in one state but mean reversion in another. A volume-shock feature may matter differently during high-dispersion and low-dispersion periods.

The paper does not attempt to estimate a complete structural model of each regime. Instead, it uses diagnostic evidence. Same-regime log-loss relative to global log-loss, cross-regime transfer behavior, and forecast relationship distance all provide partial views of relationship shift. Because each view is incomplete, the framework separates H1, H2, H3, and H4 rather than collapsing them into one claim.

### 3.4 Transferability Loss

Transferability loss means that a relationship learned or evaluated under one regime performs worse when transferred to another. Let \(i \rightarrow i\) denote same-regime reference behavior and \(i \rightarrow j\) denote cross-regime behavior. If cross-regime performance is weaker than same-regime performance, the transfer cell shows a loss. If it is similar or stronger, the transfer cell does not show loss under that metric.

The framework does not assume that all cross-regime cells must deteriorate. That would be too strong for financial data and too simple for machine-learning forecasts. Two regimes may differ in centroid distance while sharing a useful predictive relationship. Another pair may have modest state distance but sharply different forecast behavior. H3 is therefore satisfied by non-identical transfer behavior, while H4 requires a stronger distance-linked pattern.

### 3.5 Regime Distance and Forecast Relationship Distance

Regime Distance measures separation between reconstructed latent states in the space of lagged state features. Forecast Relationship Distance measures separation between regime-specific forecast relationships. RD and FRD can be useful because they turn qualitative state differences into testable diagnostic quantities. They remain diagnostic distances, not causal mechanisms.

This boundary is essential. A positive association between distance and transfer loss would not prove that distance causes transfer loss. It would show that the chosen distance measure is aligned with observed transfer behavior in the tested design. Conversely, weak distance evidence does not invalidate regime-conditional forecasting if H1 and H2 are supported. It only limits the claim that a particular distance metric explains transferability loss.

### 3.6 Diagnostic Metrics

The Regime Information Gain compares global log-loss with regime-conditioned log-loss:

$$
\mathrm{RIG}
=
\mathrm{LogLoss}_{global}
-
\mathrm{LogLoss}_{regime}
$$

Log-loss is defined as:

$$
\mathrm{LogLoss}
=
-\frac{1}{N}
\sum_{n=1}^{N}
\left[
y_n \log(\hat{p}_n)
+
(1-y_n)\log(1-\hat{p}_n)
\right]
$$

Transfer-retention ratio compares cross-regime performance with same-regime reference performance:

$$
\mathrm{TRR}_{ij}
=
\frac{
\mathrm{Accuracy}_{cross}(i \rightarrow j)
}{
\mathrm{Accuracy}_{same}(i \rightarrow i)
}
$$

Transfer gap measures the same comparison as a difference:

$$
\mathrm{TG}_{ij}
=
\mathrm{Accuracy}_{same}(i \rightarrow i)
-
\mathrm{Accuracy}_{cross}(i \rightarrow j)
$$

Regime Distance is computed as standardized centroid distance over the selected lagged state columns:

$$
\mathrm{RD}_{ij}
=
\left\|
\mu_i^{std}
-
\mu_j^{std}
\right\|_2
$$

Cosine-based Regime Distance is computed as:

$$
\mathrm{RD}^{cosine}_{ij}
=
1
-
\frac{
(\mu_i^{std})^\top \mu_j^{std}
}{
\left\|\mu_i^{std}\right\|_2
\left\|\mu_j^{std}\right\|_2
}
$$

Forecast Relationship Distance is represented generically as a distance between regime-specific forecast relationship summaries:

$$
\mathrm{FRD}_{ij}
=
d(\theta_i,\theta_j)
$$

Positive RIG indicates lower validation log-loss when regime conditioning is used. TRR below 1 indicates weaker cross-regime retention than same-regime reference performance. Positive TG indicates a transfer gap. Larger RD or RD_cosine indicates greater separation between reconstructed latent-regime centroids under the chosen distance definition. FRD summarizes differences between forecast relationships, such as coefficient distance, coefficient cosine distance, predicted probability distribution distance, Brier score gap, or calibration error gap where feasible.

## 4. Hypotheses

### 4.1 H1: Regime Information Adds Predictive Value

H1 states that reconstructed regime information adds predictive value. The theoretical rationale comes from adaptive-market reasoning: if market ecology and conditional dynamics vary over time, a state variable may help organize forecast errors and probabilities. The empirical test is validation Regime Information Gain. H1 is supported if regime-conditioned validation log-loss is lower than global validation log-loss.

The final status is diagnostically supported. Validation RIG = 0.403333. This is evidence that the reconstructed latent regimes are information-bearing in the validation design. It is not proof that the regimes are structural economic states.

### 4.2 H2: Forecasting Skill Differs Across Regimes

H2 states that forecasting skill differs across reconstructed regimes. The rationale is that if the mapping from features to future direction is conditional, then forecast correctness should not be evenly distributed across all state contexts. A model may be more accurate in states with clearer directional structure and less accurate in states with noisy or unstable relationships.

The empirical test compares correctness across reconstructed latent regimes in validation. H2 is diagnostically supported because the validation accuracy gap = 0.085666, chi-square = 121.98, and p = 3.25e-27. The result supports regime-conditional diagnostic interpretation, but it does not establish a complete conditional distribution model or a market-wide efficiency result.

### 4.3 H3: Forecasting Skill Is Not Perfectly Transferable Across Regimes

H3 states that forecasting skill is not perfectly transferable across regimes. The theoretical rationale follows from relationship shift. If a relationship is partly state-dependent, then a same-regime reference cell and a cross-regime transfer cell may behave differently. The empirical test examines TRR and TG ranges across validation regime pairs.

The final status is mixed and partially supported. Validation TRR ranges from 0.826009 to 1.169407, and validation TG ranges from -0.074978 to 0.090714. The evidence shows non-identical transfer behavior, but transfer loss is not uniform. That mixed result is useful because it indicates that transferability depends on regime pair and metric rather than following a simple universal deterioration pattern.

### 4.4 H4: Transferability Decreases as Regime Distance Increases

H4 states that transferability decreases as regime distance increases. This is the strongest hypothesis because it goes beyond state dependence and asks whether a diagnostic distance measure explains transfer loss. The theoretical rationale is plausible: regimes that are farther apart in latent-state space may have less similar forecast relationships. The empirical burden is higher because distance must align with transfer loss in the expected direction and with sufficient statistical support.

The final status is narrow partial support only. The original K=3 validation distance tests are nonsignificant. The strengthening audit reports 3 supported, 294 weak, 557 not supported, and 196 inconclusive rows. Support appears only for K=4 train-only GMM, ticker-level observations, RD_cosine, TRR_balanced_accuracy, TG_balanced_accuracy, and logloss_gap. This is not general support for H4 and does not prove a causal distance mechanism.

## 5. Data and Methodology

### 5.1 VN30 Walk-Forward Benchmark

The empirical setting is VN30 stock-level directional prediction with full 30-stock headline coverage for the main h40 benchmark row. The broader benchmark evaluates multiple horizons, including h20, h40, h60, and h80, but the primary latent-regime evidence in this paper is the h40 design. The target is directional rather than return magnitude forecasting. The paper does not evaluate a portfolio, execution process, or trading rule.

The walk-forward structure separates training, validation, and final scoring. Training data are used to fit model components and to reconstruct latent regimes. Validation data govern model, feature, threshold, and claim selection. Final-window outputs are retained only as descriptive scoring context. This structure is designed to prevent final-window observations from becoming the source of the headline claim.

The main bounded benchmark row is Logistic L2 using the C-closest reference feature set, validation-selected threshold 0.55, final accuracy 61.63%, and full 30-stock coverage. The result is intentionally stated with its selection boundary. The model family, feature set, threshold, final score, and coverage all matter because changing any one of these could change the claim.

### 5.2 Validation-Only Selection

Validation-only selection is the central design control. The paper allows validation evidence to choose the eligible model row and to support the hypothesis tests. It does not allow final-window scores to promote a row into claim eligibility. This distinction is especially important because the benchmark contains attractive descriptive rows, including router 63.33% and soft voting near 62.00%.

The router and soft-voting outputs remain useful. They can indicate that regime-aware routing or cooperation among models deserves future study. However, because they are not the bounded validation-selected h40 headline row, they are not claim-eligible evidence. They are discussed only as descriptive final-window context.

### 5.3 Leakage Controls

The leakage boundary is defined around feature timing, target timing, regime construction, and final-window use. The state features used for the primary regime reconstruction are lagged and include return, volatility, breadth, dispersion, and volume-shock context. The train-only 3-regime GaussianMixture is fitted on 259 train unique timestamps. Final-window labels and final-window scores are not used to construct the primary claim.

The paper also distinguishes future-blind validation from descriptive scoring. A future-blind claim would require pre-specified rules applied to later data without post hoc adjustment. This paper does not make that stronger claim. It reports a validation-governed diagnostic result and identifies future-blind validation as a future-work requirement.

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

### 5.4 Train-Only GMM Construction

The primary latent-regime construction uses a train-only 3-regime GaussianMixture. The term train-only is important. The GMM is fitted on training timestamps and then used to assign or evaluate regimes without letting final-window outcomes determine the latent-state structure. The state feature set is local and lagged, so the regimes are reconstructed from information that is intended to be available before the target horizon.

The resulting regimes are not assigned economic names. They are not labeled as crisis, expansion, bull, bear, sideway, or liquidity regimes. The paper avoids those labels because the GMM is a statistical reconstruction rather than an external economic classification. The interpretation is therefore diagnostic: if the reconstructed states organize errors and transfer behavior, they are useful for the paper's forecasting audit.

### 5.5 Transfer-Matrix Construction

The transfer matrix compares same-regime and cross-regime behavior. A same-regime cell \(i \rightarrow i\) provides the reference for how a forecasting relationship behaves when source and target regimes match. A cross-regime cell \(i \rightarrow j\) measures how that relationship behaves when evaluated under a different target regime. TRR expresses cross-regime performance as a ratio relative to same-regime reference performance, while TG expresses the same idea as a difference.

This matrix structure is important because it keeps H3 from becoming a vague statement about instability. The evidence is cell-specific. Some cross-regime cells can show deterioration while others show improvement. The hypothesis is therefore evaluated through the pattern of transfer behavior, not by assuming that every transfer must lose accuracy.

### 5.6 H4 Strengthening Audit

The H4 strengthening audit examines whether a distance-linked transferability interpretation survives additional designs. It includes FRD variants such as coefficient L2 distance, coefficient cosine distance, predicted probability distribution distance, Brier score gap where probabilities are available, and calibration error gap where feasible. It also examines multi-horizon diagnostics for h20, h40, h60, and h80 when labels and features are available without a heavy rerun.

Regime-count robustness is included through train-only K=2 and K=4 GMM variants where lightweight fitting is feasible. K=2 mostly produced inconclusive or degenerate-distance tests. K=4 produced the only supported H4 rows. Those supported rows appear only for ticker-level observations, RD_cosine, TRR_balanced_accuracy, TG_balanced_accuracy, and logloss_gap.

The audit uses conservative status rules. A row is supported only when it has the expected direction, p < 0.05, and consistency across at least two related metrics or two horizons. Expected-direction but nonsignificant rows are weak. Wrong-direction or near-zero relationships are not supported. Insufficient or unstable estimates are inconclusive.

### 5.7 Final-Window Evidence Boundary

Final-window results are retained because they are informative about historical scoring behavior, but their role is limited. The final-window RIG = 0.429032 is descriptive only. The final descriptive accuracy gap = 0.118453, chi-square = 44.28, and p = 2.4232e-10 is also descriptive only. These values can be compared with validation evidence, but they cannot strengthen the validation-governed claims.

This boundary prevents escalation from descriptive observations to stronger claims. It also makes the paper's contribution more transparent. The evidence can still be meaningful even when final-window results are not claim-eligible, because the hypotheses are explicitly tied to validation diagnostics and bounded robustness designs.

## 6. Results

![Figure 2. Hypothesis Evidence Status Panel. H1 and H2 are diagnostically supported, H3 is mixed/partially supported, and H4 is only narrowly partially supported.](figures/figure2_hypothesis_status_panel.png)

**Table 3. Hypothesis Evidence Summary**

| Hypothesis | Key evidence | Final status | Claim-safe wording |
| --- | --- | --- | --- |
| H1 | validation RIG = 0.403333 | diagnostically supported | H1 is diagnostically supported in validation by positive RIG; this is diagnostic evidence, not proof of economic regime existence. |
| H2 | accuracy gap = 0.085666; chi2 = 121.98; p = 3.25e-27 | diagnostically supported | H2 is diagnostically supported: validation forecast correctness differs across reconstructed latent regimes. |
| H3 | TRR range = 0.826009 to 1.169407; TG range = -0.074978 to 0.090714 | mixed / partially supported | H3 is mixed/partially supported: transfer behavior differs by regime pair, but cross-regime loss is not uniform. |
| H4 | Original K=3 validation RD tests nonsignificant; strengthening audit: 3 supported, 294 weak, 557 not supported, 196 inconclusive. | partially supported under narrow robustness design; not general support | H4 is partially supported only under a narrow robustness design using K=4 train-only GMM, ticker-level observations, cosine-based regime distance, and balanced-accuracy/log-loss transfer metrics. It is not established as a general distance-transfer law. |

### 6.1 Hypothesis-Level Summary

Table 3 summarizes the evidence. H1 and H2 are the strongest results because both are validation-based and directly tied to the primary h40 latent-regime design. H3 is informative but more nuanced because cross-regime transfer cells are heterogeneous. H4 is the most bounded because support appears only under a narrow K=4 ticker-level robustness design.

This pattern is not a failure of the framework. It separates three ideas that are often conflated. First, regime information can be useful. Second, forecast correctness can differ across regimes. Third, a particular distance metric may still fail to explain transfer loss generally. The paper supports the first two ideas, partially supports the third through H3, and supports the fourth only in a narrow robustness design.

### 6.2 H1: Regime Information Gain

H1 is diagnostically supported. Validation RIG = 0.403333, meaning regime-conditioned validation log-loss is lower than the global reference under the diagnostic construction. This indicates that the reconstructed latent regime carries information relevant to probabilistic forecast quality.

The result is meaningful because it is validation-based. It is not derived from final-window selection, and it is not based on a post hoc economic label. The regime labels are reconstructed from lagged state features. Therefore, H1 supports the statement that the train-only latent-state reconstruction is information-bearing for validation diagnostics.

The result remains bounded. Positive RIG does not prove that the regimes are structural economic states. It does not imply that regime conditioning would remain useful in all later periods. It does not imply profitability or practical deployability. It supports a narrower claim: in this VN30 h40 benchmark, regime-conditioned log-loss improves relative to the global reference in validation.

### 6.3 H2: Regime-Conditional Forecasting Skill

H2 is diagnostically supported. Forecast correctness differs across reconstructed latent regimes in validation, with accuracy gap = 0.085666, chi-square = 121.98, and p = 3.25e-27. This is strong diagnostic evidence that model correctness is not evenly distributed across the reconstructed regimes.

The interpretation is regime-conditional forecasting skill. A single global accuracy masks the fact that some reconstructed states are easier or harder for the model. This matters because a model-ranking paper could report the same global score without revealing that performance is state-dependent. The H2 result therefore adds explanatory depth to the benchmark.

**Table 4. Regime Information and Regime-Conditional Accuracy**

| Diagnostic | Value | Role | Interpretation |
| --- | --- | --- | --- |
| RIG (validation) | 0.403333 | primary | Primary diagnostic evidence: same-regime log-loss is lower than global log-loss, giving positive validation RIG. |
| RIG (final) | 0.429032 | descriptive_only | Descriptive final-window scoring only; not selection evidence and not promoted beyond context. |
| Accuracy association (validation) | gap 0.085666; chi-square 121.98; p 3.25e-27 | primary | Primary diagnostic evidence: global-model correctness differs across latent regimes in validation. |
| Accuracy association (final) | gap 0.118453; chi-square 44.28; p 2.4232e-10 | descriptive_only | Descriptive final-window scoring only; not selection evidence. |

The final-window accuracy association is directionally consistent with the idea that correctness differs across regimes, but it remains descriptive only. The claim rests on validation. This distinction preserves the paper's selection discipline and prevents final-window scoring from becoming inferential evidence.

### 6.4 H3: Transferability Across Regimes

H3 is mixed and partially supported. Validation TRR ranges from 0.826009 to 1.169407, and validation TG ranges from -0.074978 to 0.090714. These ranges show that cross-regime transfer behavior is not identical across regime pairs. Some cells lose performance relative to same-regime references, while others retain or improve performance.

The mixed pattern is useful. If every cross-regime cell deteriorated, the interpretation would be simple: regime boundaries uniformly impair transfer. Instead, the observed pattern suggests regime-pair heterogeneity. Some reconstructed states may differ in market context but still share a forecast relationship. Others may differ in a way that weakens transfer. This is exactly the type of information a transfer matrix can reveal and a single model ranking cannot.

**Table 5. Transferability Summary**

| Split | TRR range | TG range | TG signs | Interpretation |
| --- | --- | --- | --- | --- |
| validation | 0.826009 to 1.169407 | -0.074978 to 0.090714 | +3 / -3 | Cross-regime TG has mixed signs, so H3 is mixed/partially supported rather than uniformly supported. |
| final | 0.682257 to 1.263797 | -0.138643 to 0.212575 | +4 / -2 | Descriptive final-window context only; cross-regime signs are not selection evidence. |

![Figure 3. Transfer Matrix Heatmap. Validation transfer-retention ratios show mixed cross-regime behavior and are diagnostic only.](figures/figure3_transfer_matrix_heatmap.png)

The final descriptive TRR range = 0.682257 to 1.263797 and final descriptive TG range = -0.138643 to 0.212575 show that heterogeneity also appears in final scoring, but those values are not used to strengthen H3. They are reported only to maintain transparency about the final-window context.

### 6.5 H4: Distance-Linked Transferability

H4 receives only narrow partial support. The original K=3 validation RD-vs-TRR and RD-vs-TG tests are nonsignificant. The strengthening audit finds 3 supported rows, 294 weak rows, 557 not supported rows, and 196 inconclusive rows. This distribution of statuses is important because the supported evidence is small relative to the full audit space.

The supported rows appear only for K=4 train-only GMM, ticker-level observations, RD_cosine, TRR_balanced_accuracy, TG_balanced_accuracy, and logloss_gap. These rows satisfy the expected direction and p < 0.05 requirements across related metric families. They provide legitimate narrow support, but they do not establish a general distance-transfer law.

FRD variants do not strengthen H4. They report 0 supported, 15 weak, 27 not supported, and 14 inconclusive tests. Multi-horizon tests also do not strengthen H4, reporting 0 supported, 256 weak, 458 not supported, and 56 inconclusive tests. K=2 mostly produced inconclusive or degenerate-distance tests. This pattern constrains the conclusion.

**Table 6. H4 Strengthening Audit Summary**

| Scope | Supported | Weak | Not supported | Inconclusive | Claim-safe wording |
| --- | --- | --- | --- | --- | --- |
| overall | 3 | 294 | 557 | 196 | H4 is partially supported under specific robustness designs, not established as a general law. |
| regime_count | 3 | 23 | 72 | 126 | H4 is partially supported under specific robustness designs, not established as a general law. |
| frd | 0 | 15 | 27 | 14 | H4 is partially supported under specific robustness designs, not established as a general law. |
| multihorizon | 0 | 256 | 458 | 56 | H4 is partially supported under specific robustness designs, not established as a general law. |

*Note.* Supported rows appear only for K=4 train-only GMM, ticker-level observations, RD_cosine, TRR_balanced_accuracy, TG_balanced_accuracy, and logloss_gap. FRD and multi-horizon tests do not reach supported status.

![Figure 4. H4 Robustness Count Panel. The strengthening audit has 3 supported rows, while most tests are weak, not supported, or inconclusive.](figures/figure4_h4_robustness_count_panel.png)

The claim-safe conclusion is therefore precise. H4 is partially supported only under a narrow K=4 ticker-level robustness design and is not general support. RD and FRD remain diagnostic distances, not causal mechanisms. The evidence suggests that distance may matter under specific reconstruction and metric choices, but the paper does not claim a broad monotonic relationship between distance and transfer loss.

### 6.6 Main h40 Benchmark Result and Descriptive Final-Window Context

The main h40 benchmark result remains Logistic L2 using the C-closest reference feature set, validation-selected threshold 0.55, final accuracy 61.63%, and full 30-stock coverage. This result is the bounded claim-eligible score because it is tied to validation-governed selection and complete VN30 coverage.

The router 63.33% final-window score is not claim-eligible. It is useful as descriptive context because it suggests that routing ideas may be worth future-blind evaluation, but it does not replace the main h40 result. Soft voting near 62.00% is also not claim-eligible. It can motivate further research into model cooperation, but it cannot be promoted into a headline claim.

This separation is central to the paper's reliability. A benchmark can contain descriptive observations that are scientifically interesting while still keeping the formal claim boundary narrow. The paper reports those observations transparently and assigns them the correct inferential status.

## 7. Discussion

The results support a regime-aware diagnostic view of VN30 machine-learning forecasts. H1 and H2 show that reconstructed latent regimes are not merely decorative labels. They are associated with improved validation log-loss and with differences in forecast correctness. This supports the core idea that forecast performance is conditional on market-state context.

At the same time, H3 and H4 prevent overstatement. Transfer behavior differs across regime pairs, but the direction is mixed. Distance-linked transfer decay receives support only under a narrow K=4 ticker-level robustness design. These findings suggest that regime-aware forecasting is informative but not reducible to a simple rule that farther regimes always produce weaker transfer.

### 7.1 Theoretical Interpretation of Partial Support

The pattern H1 supported, H2 supported, H3 mixed and partially supported, and H4 narrowly partially supported should be interpreted as partial theoretical support rather than as rejection of the framework. H1 and H2 address the core theoretical objective: whether reconstructed market-state information matters for forecast diagnostics. Both are supported in validation. This means the framework succeeds in showing that forecast behavior is not fully invariant across reconstructed regimes.

H1 indicates that same-regime validation log-loss improves relative to the global reference. This supports the idea that the reconstructed state variable contains information about forecast quality. The evidence does not require the regimes to be named economic states. A latent regime can be diagnostically useful even if its economic interpretation remains incomplete.

H2 strengthens the theoretical point by showing that forecast correctness differs across regimes. This is the clearest evidence that a single global benchmark score is incomplete. The same model can behave differently across reconstructed market states. That finding extends adaptive-market reasoning into machine-learning forecast evaluation by showing that state dependence appears not only in return predictability arguments but also in the realized correctness of an ML benchmark.

H3 adds nuance. Mixed transfer behavior means that some regime pairs are less transferable while others are not. This does not invalidate the framework. It shows that transferability is a matrix, not a scalar. A theory of regime-dependent forecasting should allow some states to share predictive structure and others to diverge. The mixed result therefore opens a richer research agenda focused on source-target regime pairs, metric sensitivity, and model-specific transfer behavior.

H4 is the narrowest result. The K=4 ticker-level RD_cosine evidence suggests that a distance-linked interpretation can hold under specific conditions, but the broader audit does not support a general distance-transfer rule. This is a useful boundary. It prevents the paper from turning a plausible diagnostic mechanism into an unsupported causal claim. It also identifies where future work should focus: pre-specified distance metrics, independent future windows, external state labels, and richer measures of forecast relationship shift.

Taken together, the evidence extends AMH-style reasoning into ML forecast diagnostics without overclaiming. AMH motivates time-varying predictability. This paper shows that, in the VN30 h40 validation design, reconstructed regimes are associated with probabilistic forecast improvement and correctness differences. It then shows that transferability is heterogeneous and only partly explained by the tested distance measures. That combination is more informative than a simple success-or-failure model ranking.

Partial support also helps define the framework's scientific value. A framework that only reports fully supported hypotheses can hide where its assumptions fail. Here, the unsupported and weak H4 rows are part of the contribution because they show that distance metrics must be treated carefully. They also show that FRD variants and multi-horizon tests do not automatically confirm the K=4 RD_cosine pattern.

The results therefore satisfy the core theoretical objective. The objective was not to prove a universal transfer-loss law. It was to test whether regime-aware diagnostics reveal state-dependent forecast behavior and bounded transferability patterns in an emerging equity market benchmark. H1 and H2 support that objective directly. H3 and H4 refine it by showing where transferability is heterogeneous and where distance explanations remain limited.

![Figure 5. RD/FRD Claim Boundary Diagram. Diagnostic distances are not causal mechanisms, and descriptive final-window outputs do not create trading, profitability, or live-deployment claims.](figures/figure5_rd_frd_claim_boundary.png)

### 7.2 Why Model-Ranking Evidence Is Insufficient

The results illustrate why model rankings alone are insufficient for this research question. A ranking can identify the best row under a chosen metric, but it cannot explain whether that row is robust across reconstructed regimes. It also cannot reveal whether correctness is concentrated in particular states or whether cross-regime transfer behaves symmetrically.

The regime-transferability audit adds interpretive structure. H1 asks whether regime conditioning improves probabilistic diagnostics. H2 asks whether correctness differs across regimes. H3 asks whether cross-regime cells retain same-regime behavior. H4 asks whether diagnostic distances explain transfer loss. These questions turn a benchmark from a leaderboard into a map of conditional performance.

This matters because financial machine-learning studies are vulnerable to attractive but unstable results. A model can score well in a final period because that period resembles the validation state or because one regime dominates the sample. Without regime diagnostics, the researcher may mistake conditional success for general stability. The paper avoids that escalation by reporting final-window context while keeping claims tied to validation.

### 7.3 Interpretation for Emerging-Market Forecasting

The emerging-market context makes the findings especially relevant but does not enlarge the claim. Liquidity, turnover, participation, and market frictions can change the statistical environment in which forecasts are made. The VN30 evidence is consistent with the idea that forecast behavior is conditional on reconstructed market-state context. However, consistency with emerging-market intuition is not the same as proof of a broad emerging-market law.

The paper therefore uses emerging-market literature as motivation and boundary. It motivates why state instability is worth studying. It also warns against overgeneralization because emerging markets differ across institutions, liquidity, investor base, and data quality. The VN30 result remains a VN30 stock-level diagnostic result.

### 7.4 Interpretation of Final-Window Context

Final-window outputs help readers understand the historical scoring environment, but they are not inferential anchors. The final-window RIG, final accuracy association, router score, and soft-voting score are all descriptive. They are reported because hiding them would reduce transparency, but promoting them would violate the selection boundary.

This distinction is important for future work. Descriptive final-window patterns can motivate future-blind tests. They can suggest which routers, ensembles, or distance metrics deserve pre-specification. They cannot retroactively turn exploratory observations into claim-eligible evidence.

## 8. Limitations and Future Work

### 8.1 Latent Regime Reconstruction

The first limitation is latent regime reconstruction. The regimes are generated by a train-only 3-regime GaussianMixture using lagged state features. They are not externally validated economic labels. This means the regimes should be interpreted as reconstructed diagnostic states. They may capture combinations of volatility, breadth, dispersion, return, and volume-shock context, but the paper does not assign definitive economic meanings to them.

Future work should compare reconstructed regimes with external state labels. Potential external labels could include macroeconomic periods, liquidity shocks, market-wide stress episodes, policy-related episodes, or independently defined volatility states. Such comparisons would help determine whether diagnostic regimes correspond to interpretable economic environments.

### 8.2 Final-Window Scoring

The second limitation is final-window scoring. Final-window outputs are useful for descriptive context, but they cannot serve as selection evidence. This limits the strength of the paper's claims, deliberately. The paper does not claim that final-window router or soft-voting scores define a better system. It reports them as context only.

Future work should use future-blind validation. A future-blind design would pre-specify the model family, feature set, threshold, regime construction, distance metrics, transfer metrics, and claim rules before observing later outcomes. That design would be necessary to move from diagnostic historical evidence to stronger prospective evidence.

### 8.3 H4 Distance Metrics

The third limitation is distance measurement. RD, RD_cosine, and FRD variants are diagnostic summaries. They may omit important aspects of the market state or forecast relationship. A centroid distance can miss nonlinear structure. A coefficient distance can be unstable when features are correlated. Probability-distribution distance can depend on calibration and sample composition. Balanced-accuracy transfer metrics can reveal one pattern while raw accuracy reveals another.

The H4 result reflects these limitations. The supported evidence appears only under K=4 ticker-level RD_cosine and related balanced-accuracy or log-loss metrics. FRD variants have 0 supported rows. Multi-horizon tests have 0 supported rows. These facts show that H4 remains an open question, not a settled mechanism.

Future work should pre-specify richer distance families and evaluate them in independent windows. It should also examine whether distance metrics behave differently by algorithm. Logistic models, tree ensembles, nearest-neighbor models, and neural models may have different sensitivities to state shifts.

### 8.4 Sample, Horizon, and Model Boundaries

The paper focuses on the h40 primary latent-regime design. Multi-horizon diagnostics are included in the H4 audit, but they do not produce supported H4 rows. This limits broad claims across forecast horizons. A relationship that appears at h40 may not appear at h20, h60, or h80.

The model boundary is also important. The main claim-eligible row is Logistic L2 with the C-closest reference feature set. The paper does not claim that all model families behave similarly. Algorithm-specific sensitivity is a natural future-work direction because different learners may respond differently to regime shifts, feature instability, and class balance.

### 8.5 Additional Future Work

Future work should evaluate liquidity shocks, sentiment, macro variables, and external economic regime labels. Liquidity shocks could help explain when transferability weakens because market depth and transaction conditions change. Sentiment variables could capture participation or attention effects not visible in price-volume features. Macro variables could connect reconstructed regimes to broader economic conditions.

Future work should also examine whether the transfer matrix changes through time. A regime pair that transfers well in one period may not transfer well later. That possibility is consistent with adaptive-market reasoning and would require repeated future-blind windows rather than a single historical audit.

Finally, future work should separate diagnostic usefulness from operational usefulness. A diagnostic framework can improve understanding even when it is not ready for deployment. Moving toward operational use would require additional evidence on transaction costs, slippage, liquidity, risk, turnover, stability, monitoring, and governance. Those topics are outside this paper's claim boundary.

## 9. Conclusion

This paper provides a bounded VN30 regime-transferability audit. H1 is diagnostically supported because validation RIG = 0.403333. H2 is diagnostically supported because forecast correctness differs across reconstructed regimes, with accuracy gap = 0.085666, chi-square = 121.98, and p = 3.25e-27. H3 is mixed and partially supported because transfer behavior differs across regime pairs, with validation TRR = 0.826009 to 1.169407 and TG = -0.074978 to 0.090714. H4 is partially supported only under a narrow K=4 ticker-level robustness design and is not general support.

The main bounded h40 result remains Logistic L2 using the C-closest reference feature set, validation-selected threshold 0.55, final accuracy 61.63%, and full 30-stock coverage. Router 63.33% and soft voting near 62.00% remain descriptive only and are not claim-eligible.

The theoretical contribution is a regime-aware diagnostic framework for machine-learning forecast transferability in an emerging equity market. The empirical contribution is a claim-bounded VN30 audit showing that reconstructed regimes help explain validation performance and transfer heterogeneity. The boundary is equally important: the paper does not claim causal distance mechanisms, trading profitability, investment relevance, production deployment, live-deployment readiness, unreported score-target performance, or broad generalization.

## Appendix A: Claim Boundary

This appendix restates the claim boundary so the expanded paper remains readable without changing the empirical interpretation.

Benchmark evidence is discovery and diagnostic only. The benchmark organizes model, feature, threshold, regime, and transfer evidence, but it does not prove that the reconstructed regimes are structural economic states. It also does not prove market inefficiency, profitability, or deployable forecasting value.

H1 is diagnostically supported. The support comes from validation Regime Information Gain, not from final-window promotion. H2 is diagnostically supported because validation forecast correctness differs across reconstructed regimes. H3 is mixed and partially supported because transfer behavior differs across regime pairs but transfer loss is not uniform. H4 is partially supported only under a narrow K=4 ticker-level robustness design and is not general support.

RD and FRD are diagnostic distances. They summarize regime or forecast-relationship separation under selected definitions. They are not causal mechanisms. A supported distance association would be evidence of alignment between distance and transfer loss, not proof that distance causes transfer loss.

Final-window outputs are descriptive scoring-only. Final RIG = 0.429032, final descriptive accuracy gap = 0.118453, chi-square = 44.28, and p = 2.4232e-10 are reported as context. Router 63.33% and soft voting near 62.00% are also descriptive only. They are not claim-eligible.

The main bounded h40 result remains Logistic L2 using the C-closest reference feature set, validation-selected threshold 0.55, final accuracy 61.63%, and full 30-stock coverage. Vietnam AMH evidence is contextual motivation only and does not replace this paper's VN30 stock-level evidence. International AMH evidence motivates time-varying predictability only and does not prove VN30 H4.

The paper makes no trading, profitability, investment, production-deployment, live-deployment, unreported score-target, or broad generalization claim.

## Appendix B: Metric Definitions

RIG measures the reduction in log-loss from using regime-conditioned diagnostics rather than a global reference. A positive value means the regime-conditioned diagnostic has lower log-loss. In this paper, validation RIG is primary and final RIG is descriptive.

Log-loss evaluates probabilistic forecast quality. It penalizes confident incorrect probabilities more heavily than less confident errors. This makes it useful for regime diagnostics because a regime can improve probability quality even when accuracy changes less visibly.

TRR measures transfer retention. A value below 1 indicates that cross-regime accuracy is lower than same-regime reference accuracy. A value above 1 indicates that cross-regime accuracy is higher than the same-regime reference for that cell. TRR should be interpreted by regime pair rather than as a universal property.

TG measures the transfer gap. A positive TG indicates that same-regime accuracy exceeds cross-regime accuracy. A negative TG indicates that cross-regime accuracy exceeds the same-regime reference. The validation TG range of -0.074978 to 0.090714 is why H3 is mixed rather than uniformly supported.

RD measures standardized latent-regime centroid distance over lagged state columns. RD_cosine measures angular separation between standardized centroids. FRD measures forecast relationship distance, with variants based on coefficients, predicted probability distributions, Brier score gaps, or calibration error gaps where feasible.

These metrics answer different questions. RIG addresses information gain. Accuracy association addresses correctness differences. TRR and TG address transfer behavior. RD and FRD address whether distance measures align with transfer loss. The paper keeps those interpretations separate to avoid overstating any single metric.

## Appendix C: Hypothesis-to-Evidence Map

H1 maps to validation Regime Information Gain. Evidence: validation RIG = 0.403333. Status: diagnostically supported. Interpretation: reconstructed regimes improve validation log-loss relative to the global reference.

H2 maps to validation correctness association. Evidence: accuracy gap = 0.085666, chi-square = 121.98, p = 3.25e-27. Status: diagnostically supported. Interpretation: forecast correctness differs across reconstructed regimes.

H3 maps to transfer-matrix behavior. Evidence: validation TRR = 0.826009 to 1.169407 and validation TG = -0.074978 to 0.090714. Status: mixed and partially supported. Interpretation: cross-regime transfer behavior is heterogeneous and not uniformly worse.

H4 maps to distance-transfer tests. Evidence: 3 supported, 294 weak, 557 not supported, and 196 inconclusive rows in the strengthening audit. Status: partially supported only under a narrow K=4 ticker-level robustness design. Interpretation: distance-linked transferability is plausible in one bounded design but not established generally.

Final-window evidence maps to descriptive context only. Evidence includes final RIG = 0.429032, final descriptive accuracy gap = 0.118453, chi-square = 44.28, p = 2.4232e-10, final TRR = 0.682257 to 1.263797, and final TG = -0.138643 to 0.212575. Status: descriptive only. Interpretation: useful for context, not claim escalation.

## Appendix D: H4 Robustness Interpretation

The H4 audit is intentionally conservative because distance-linked transferability is the strongest claim. It requires more than evidence that regimes differ. It requires evidence that larger diagnostic distance is associated with weaker transfer in the expected direction, with statistical support and consistency.

The original K=3 validation design does not support H4. This matters because K=3 is the primary latent-regime construction for the paper. The strengthening audit adds robustness designs, but those designs do not replace the primary boundary. They can only provide bounded additional evidence.

K=4 is the only design that produces supported rows. The support is limited to ticker-level observations, RD_cosine, TRR_balanced_accuracy, TG_balanced_accuracy, and logloss_gap. This suggests that angular separation among K=4 reconstructed centroids may capture a transfer-relevant aspect of the state space. However, because the support is narrow, it cannot be generalized to all distances, all regime counts, all horizons, or all transfer metrics.

FRD variants produce 0 supported, 15 weak, 27 not supported, and 14 inconclusive rows. This indicates that forecast relationship distance, as operationalized in the audit, does not provide supported H4 evidence. Multi-horizon tests produce 0 supported, 256 weak, 458 not supported, and 56 inconclusive rows. This indicates that H4 does not generalize across the tested horizons.

The correct interpretation is bounded partial support. H4 remains theoretically plausible and empirically incomplete. Future work should pre-specify distance metrics, test them in future-blind windows, and compare them with external economic state labels.

## Appendix E: Table and Figure Notes

The expanded paper keeps the original seven tables and five figures. No new empirical tables or figures are introduced. This preserves the evidence pack and avoids creating unsupported numerical claims.

Figure 1 summarizes the research design flow. It should be read as a workflow diagram rather than evidence. Figure 2 summarizes hypothesis status. It reflects the claim boundary: H1 and H2 are diagnostically supported, H3 is mixed and partially supported, and H4 is narrowly partially supported. Figure 3 visualizes transfer-matrix behavior and supports the interpretation that H3 is heterogeneous. Figure 4 visualizes the H4 robustness count pattern. Figure 5 states the RD/FRD claim boundary.

Table 1 states the study design and claim boundary. Table 2 documents the latent-regime construction audit. Table 3 maps hypotheses to evidence and final status. Table 4 reports regime information and regime-conditional accuracy. Table 5 reports transferability ranges. Table 6 reports H4 strengthening audit counts. Table 7 maps the 22 verified references to their role in the paper.

The tables and figures should not be read as expanding the claim beyond the text. The table and figure notes are included to prevent common misreadings: descriptive final-window rows are not selection evidence, distance metrics are not causal mechanisms, and the H4 support is not general.

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
