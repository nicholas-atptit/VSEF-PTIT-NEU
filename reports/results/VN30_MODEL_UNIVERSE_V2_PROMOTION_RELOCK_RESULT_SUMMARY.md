# VN30 Model Universe V2 Promotion Relock Result Summary

## Required Answers

1. What produced the exploratory 69.86% absolute-direction final row: the strongest non-trivial comparable h40 source was `direction__naive_bayes__absolute_direction__h40__market_context` with final accuracy 69.86%, validation accuracy 55.57%, and validation lift +2.00 pp. The absolute highest h40 final row was `direction__linear_svm__absolute_direction__h40__combined_strategy_features` at 70.29%, but it did not pass validation-lift promotion screening.
2. What produced the exploratory 74.57% market-relative final row: `direction__always_down__market_relative_vn30__h40__all_safe_features` with final accuracy 74.57%; this is a trivial/simple baseline pattern with validation lift +0.00 pp, so it is not promotable.
3. Were those results isolated one-offs or cluster-supported: the absolute-direction source is `cluster_supported` and the market-relative source is `cluster_supported` under the model/target/horizon/feature cluster audit.
4. Could any family be re-locked by validation: yes, V2 froze final-discovered hypothesis families and selected exact candidates by validation only, but all such relocks remain future-blind-required because the families were discovered from final exploratory rows.
5. Did any relocked direction candidate beat 61.61% on comparable scope: true for `direction__linear_svm__absolute_direction__h40__market_context` with final accuracy 69.86%; this is future-blind-required and not a replacement claim. The validation-best absolute h40 relock was `direction__hist_gradient_boosting__absolute_direction__h40__relative_strength` with final accuracy 55.14%.
6. Did any relocked market-relative candidate beat 64.44% on comparable scope: false for non-trivial relocks. The strongest row over 64.44% was `direction__always_down__market_relative_vn30__h40__all_safe_features` at 74.57%, but its label is `not_claimable`.
7. Did any relocked price/return candidate beat random walk/last price on final: false for validation-supported relocks. Best validation-supported final improvement was `price__simple_relative_strength__market_excess_return_h__h20__compact_stable_features` with final RMSE improvement -0.76 pp.
8. Which results remain exploratory: all source final-ranked rows and all relocked rows remain diagnostic/future-blind-required unless confirmed by a new pre-registered future-blind run.
9. Exact claim boundary: offline diagnostic-only; no trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, VN100, index-as-stock, tag, merge, push --mirror, DOCX, or production claim is made.

## Relocked Direction Candidates

- Best absolute h40 relock: `direction__hist_gradient_boosting__absolute_direction__h40__relative_strength`; validation 59.43%, final 55.14%, label `future_blind_required`.
- Strongest absolute h40 future-blind-required final diagnostic: `direction__linear_svm__absolute_direction__h40__market_context`; validation 56.43%, final 69.86%, label `future_blind_required`.
- Best market-relative h40 relock: `direction__simple_relative_strength__market_relative_vn30__h40__all_safe_features`; validation 62.00%, final 53.29%, label `exploratory_not_claimable`.

## Relocked Price/Return Candidate

- Best price/return relock: `price__simple_relative_strength__market_excess_return_h__h20__compact_stable_features`; validation RMSE 0.0243429, final RMSE 0.127596, final baseline improvement -0.76 pp, label `future_blind_required`.

## Audit Note

V2 uses V1 aggregate artifacts for promotion relock. Early/late validation window rows are marked unavailable when row-level validation predictions were not preserved by V1; final quarter diagnostics are included where V1 stability artifacts exist.
