# Experiments You Still Need to Run (Before Final Submission)

The revised paper is defensible as-is because every missing experiment is now
honestly acknowledged in Section VII (Limitations). But running the items below
and inserting the numbers will convert the reviewer's "Major Revision" into an
accept. They are ordered by impact-per-effort.

**IMPORTANT: Do NOT invent numbers. Every value in the revised paper comes from
your verified evaluation report. Only add data you actually measure.**

---

## 1. Latency table (ms/sample) — EASY, ~1 hour ⭐ highest priority
Reviewer asked for actual numbers, not just "16×".

Run each model on the same 500-chain test set and record wall-clock per chain:

| Model | n=10 (ms) | n=25 (ms) | n=50 (ms) | GPU batch=256 (ms/chain) |
|---|---|---|---|---|
| Python DP | | | | — |
| GraphMCMNet | | | | |
| PointerMCMNet | | | | |
| XGBoost (incl. feature extraction) | | | | — |

- Use `time.perf_counter()`, warm up 10 runs, report mean ± std over 100 runs.
- Add as "Table: Measured Inference Latency" in Section VI-D; then you can
  delete the "Speedup measurement granularity" limitation.

## 2. Greedy heuristic baselines — EASY, ~30 min
You ALREADY compute 4 greedy heuristics as features (left-to-right,
right-to-left, minimum-first, balanced). Just evaluate them standalone on the
500 test chains and add a row each to Table 2 (MAPE, median, P95, validity —
validity will be 100% since they are real parenthesizations). This directly
answers "Missing Experiment 2" and strengthens the story: structural neural
models beat both unconstrained regression AND classical heuristics.

## 3. Hu–Shing benchmark — MEDIUM, ~1–2 days
- Implement or adapt an existing Hu–Shing O(n log n) implementation
  (reference implementations exist in academic repos).
- Benchmark single-chain latency at n = 10/25/50/100 and batched throughput.
- Add the measured row to Table 5 and soften/remove the "not benchmarked" note.
- Expected honest outcome: Hu–Shing wins single-chain latency; your models win
  batched GPU throughput. That is a publishable, honest result — write it that way.

## 4. Confidence intervals — EASY, ~1 hour
You have per-chain errors in `scratch/full_eval_results_v2.json`.
- Bootstrap (10,000 resamples) 95% CIs for each model's MAPE.
- Report as e.g. "0.029% [0.021, 0.038]" in Table 2.

## 5. Scalability probe (n=100, 200, 500) — MEDIUM, ~half day
- Generate small test sets at n=100/200/500 (DP ground truth at n=500 is
  1.25×10⁸ operations — still feasible in optimized numpy/C).
- Report GNN MAPE + peak GPU memory at each n. Even if it fails at n=500 due
  to memory, reporting WHERE it fails is valuable and matches Section VII.

## 6. Training convergence curves — TRIVIAL if you logged losses
- One figure: train/val loss vs. epoch for both neural models, with curriculum
  stage boundaries marked. Add as Fig. 6.

## 7. Multi-seed variance — EXPENSIVE (~25 GPU-h per seed), optional
- Retrain GNN with 2 additional seeds; report mean ± std MAPE.
- If infeasible before the deadline, the limitation is already disclosed.

## 8. Real-world chains — HARDEST, target for camera-ready / journal extension
- Extract matrix-chain / tensor-contraction dimension sequences from TVM
  tutorials or HuggingFace transformer configs (e.g., attention projection
  chains: [batch·seq, d_model] × [d_model, d_head] × ...).
- Even a 50-chain real-world set with GNN accuracy reported would neutralize
  the reviewer's biggest external-validity objection.

## Figure fix (reviewer point on Fig. 1)
- Replace the generic system diagram with a concrete example: draw the actual
  sub-problem graph for n=4 (10 nodes, DP child edges + same-length edges)
  with the node feature vector annotated. This directly answers "Need actual
  graph example."

## Code release
- Put training/eval code in a public GitHub repo, add the link in Section V.
- Include: dataset generation script, seeds, requirements.txt, trained
  checkpoints if size permits.

---

## After running 1, 2, and 4 (≈ half a day of work):
Update these spots in `revised_paper.tex`:
- Section VI-D: insert latency table, remove "aggregate speedup" hedging
- Table 2: add greedy heuristic rows + CI brackets
- Section VII: delete the corresponding limitation paragraphs
