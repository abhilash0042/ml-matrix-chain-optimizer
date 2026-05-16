# Research Report: Comparative Analysis of MCM Optimization Models

## Executive Summary
This report details the comparative performance of **Pointer Networks** versus **Tree Ensembles (XGBoost/Random Forest)** for the Matrix Chain Multiplication optimization problem. The core finding is that structural awareness is mandatory for mathematical reliability.

## 1. The Validity Gap
The most critical finding was the "Validity Rate." A predicted cost is only valid if it is $\ge$ the true DP minimum.
- **PointerMCMNet**: 100% Validity Rate.
- **Tree Ensembles**: ~14% Validity Rate.

Statistical models routinely predict "impossible" costs because they lack a sense of the recursive multiplication constraints.

## 2. Distribution Performance
| Distribution | Pointer MAPE | XGBoost MAPE |
| :--- | :--- | :--- |
| Uniform | 0.59% | 10.29% |
| Spiky | 0.47% | 22.75% |
| Monotone | 0.02% | 16.78% |
| Bottleneck | 45.77% | 7.78% |

## 3. The Bottleneck Anomaly
While the Pointer Network dominates in almost every category, it struggled with "Bottleneck" chains (where one tiny dimension is hidden among massive ones). This is likely due to the Transformer's soft attention "blurring" the significance of the single bottleneck feature. XGBoost's hard decision splits were more effective at catching this specific outlier.

## 4. Conclusion
For systems like JIT compilers, **Pointer Networks** are the superior choice because they are "Safe by Design." They guarantee a valid parenthesization, whereas statistical models risk causing system crashes by predicting impossible performance bounds.
