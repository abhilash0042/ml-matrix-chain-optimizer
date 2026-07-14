# Response to Reviewers — Point-by-Point

Manuscript: "Structural Reasoning versus Statistical Estimation for Learned Matrix Chain Optimization"
(previously: "Constant-Time Neural Matrix Chain Optimization")

We thank the reviewer for the thorough and constructive assessment. Below we respond to each point and indicate exactly where the revision addresses it.

---

## High-Priority Revisions

### R1. "Remove 'Constant-Time' — the method is O(n²K), not constant time. This is the biggest issue."
**DONE.**
- Title changed to *"Structural Reasoning versus Statistical Estimation for Learned Matrix Chain Optimization"* — no complexity claim in the title.
- Every instance of "constant-time" / "near constant-time" removed from the abstract, introduction, and conclusion.
- Section VI-D now states explicitly: *"We emphasize that neural inference is not constant time: it scales as O(n²K) for the GNN (constant inference depth) and O(n²) for the Pointer Network."*
- The advantage is now correctly framed as (i) lower asymptotic order than O(n³) DP and (ii) vectorized batched throughput.

### R2. "Compare against Hu–Shing experimentally."
**PARTIALLY ADDRESSED — honestly scoped.**
- We did not run a Hu–Shing implementation in this revision cycle. Rather than omit the issue, the revision now (a) adds Hu–Shing as an explicit row in the complexity table (Table 5) marked "not benchmarked", (b) adds a dedicated paragraph in Section VI-D explaining that for single-chain latency at large n, Hu–Shing would likely dominate all learned approaches, and that the relevant regime for our method is *amortized batched throughput* on accelerators, and (c) lists the Hu–Shing benchmark as the top item in the new "Limitations and Threats to Validity" section (VII).
- Accuracy/validity conclusions are unaffected because Hu–Shing and DP return the same optimal cost.
- (See `experiments_still_needed.md` — this benchmark is queued for the final revision.)

### R3. "Add real datasets."
**ACKNOWLEDGED AS LIMITATION; claims re-scoped.**
- All generalization claims are now explicitly conditioned on the synthetic benchmark ("On our synthetic benchmark, ..." in the abstract; "We deliberately restrict our claims to the synthetic benchmark studied here" in the introduction).
- Section VII (Limitations) names real-world validation on compiler traces (e.g., TVM tensor programs) as the most important item of future work.

### R4. "Add latency measurements (ms/sample table)."
**PARTIALLY ADDRESSED.**
- The measured speedup figures (>16× CPU vs Python DP at n=50; >100× GPU-batched; ~12× tree ensembles) are now reported with explicit measurement conditions in Table 5 footnotes.
- A per-model millisecond latency table is acknowledged as missing in Section VII ("Speedup measurement granularity") and queued (see checklist).

### R5. "Add a theorem for the validity guarantee."
**DONE.**
- New **Theorem 1 (Structural Validity Guarantee)** with a full proof by strong induction in Section III, including the equality condition (predicted cost equals DP optimum iff predicted splits are optimal on every reachable sub-chain).
- A remark notes the guarantee is *architectural, not learned* — it holds even for an untrained model.

### R6. "Add ablation (graph edges, curriculum, auxiliary loss, feature engineering)."
**PARTIALLY DONE.**
- Section VI-D now presents a structured three-part ablation derived from the controlled comparisons already in the experimental design: (1) graph encoder vs. sequential encoder (5.1× MAPE degradation), (2) split prediction vs. cost regression (validity 100% → 18–22%), (3) length extrapolation (OOD error lower than in-distribution).
- Finer-grained component ablations (individual edge types, curriculum stages, auxiliary loss weight) are explicitly flagged as future work rather than claimed.

### R7. "Add scalability (n=100/200/500)."
**ACKNOWLEDGED AS LIMITATION.**
- Section VII ("Scale") now discusses the O(n²) node / O(n³) edge growth of the sub-problem graph, identifies memory as the binding constraint for n ≫ 100, and sketches the required mitigation (hierarchical / sparsified graph construction).

---

## Claims Moderation (reviewer's "reviewer bait" list)

| Original claim | Revised to |
|---|---|
| "Constant-Time Neural Matrix Chain Optimization" (title) | Title contains no complexity claim; body says "O(n²K) inference, constant inference depth" |
| "fundamentally superior solution path" | "the decisive design choice for learned MCM approximation" (scoped to benchmark) |
| "bypasses DP" / "near constant-time resolution" | "high-fidelity learned approximation of the DP solver" |
| "exceeds statistical baselines by three orders of magnitude" | "a three-order-of-magnitude accuracy gap **on our benchmark**" |
| "physically impossible" | "infeasible" / "mathematically invalid" (precise terminology) |
| "drop-in DP replacement" | Removed; Section VII states the models "approximate the DP solver; they do not replace it in settings requiring certified optimality" |
| "dramatically", "consistently outperformed" | Neutralized throughout |

---

## Statistical Analysis

- **Full Wilcoxon table added** (Table 4): W statistics, exact p-values, and Cohen's d for all six pairwise comparisons (previously only summarized in prose).
- **Multiple-comparison correction added**: Section V notes the Bonferroni-corrected threshold (α = 0.05/6 ≈ 0.0083) and that all comparisons survive it.
- **Distributional error statistics added** (Table 3): median error, P95 error, max error, Spearman ρ, and within-1%/5%/10% threshold accuracy for all four models — previously unreported.
- **Win rates added**: GNN wins 100% of chains vs. both tree baselines, 91% vs. Pointer Network.
- Per-seed confidence intervals and convergence curves are acknowledged as missing (Section VII, "Single training run") and queued.

---

## Writing, References, Reproducibility

- **Broken citations fixed**: all `[?]` references now resolve (Godbole, Hu–Shing, Cormen, TVM, Neo, etc.).
- **Recent literature (2021–2024) added**: neural algorithmic reasoning (Veličković & Blundell 2021; CLRS benchmark, ICML 2022; generalist algorithmic learner, LoG 2022; Transformers meet NAR, 2024) and graph transformers (GPS, NeurIPS 2022; Attending to Graph Transformers, TMLR 2024). A new "Neural Algorithmic Reasoning" paragraph in Related Work connects our split-table prediction to this line and uses it to explain the OOD result.
- **Hyperparameter selection explained**: loss weighting (0.9/0.1 selected from {0.5, 0.7, 0.9} on validation), curriculum motivation, early stopping, seeds.
- **Reproducibility details**: fixed data-split seeds, evaluation seed 42, GPU spec (RTX 4050, 6 GB), parameter counts (3.7M / 5.4M), ~25 GPU-hours, AMP training.
- **Pointer Network OOD caveat added**: the paper now discloses that the Pointer Network was trained on the full length range, so its length-41–50 comparison measures architectural robustness, not strict OOD extrapolation.
- **Baseline fairness note added**: the text now points out that the 145-feature vector *includes the outputs of four greedy MCM heuristics*, so the regression baselines received strictly more algorithmic prior knowledge than raw dimensions.

---

## Structural Changes Summary

| Section | Change |
|---|---|
| Title | Complexity claim removed |
| Abstract | Rewritten: moderated claims, added median/threshold framing, scoped to synthetic benchmark |
| I. Introduction | Contribution list rewritten; explicit claim-scoping paragraph added |
| II. Related Work | New "Neural Algorithmic Reasoning" subsection; 6 new references |
| III. Formulation | New Theorem 1 + proof + remark |
| IV. Methodology | Hyperparameter/curriculum justification; baseline-fairness note; reproducibility details |
| V. Experimental Setup | New standalone section: protocol, metrics, Bonferroni correction |
| VI. Results | Two new tables (error distribution/thresholds; full Wilcoxon); honest complexity discussion; Hu–Shing paragraph; structured ablation |
| VII. Limitations | New standalone section: six explicitly stated threats to validity |
| VIII. Conclusion | Moderated; "bypass"/"replacement" language removed |
| References | Broken refs fixed; 6 recent (2021–2024) refs added |
