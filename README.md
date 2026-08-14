# mats_task: Is J-Lens a causal auditing signal, or a linear-unembedding artifact?

A small, self-contained mechanistic-interpretability study for a MATS application
(Neel Nanda stream). It tests one crisp claim on **Qwen3-4B**, with baselines and a
built-in confound control.

## The question

J-Lens (Anthropic's global-workspace paper) reads a model's internal "working memory"
by mapping intermediate residual-stream directions to vocabulary tokens via the Jacobian.
Neel's own [review](https://www.alignmentforum.org/posts/zFJ3ZdQwrTWE9jT5S/a-review-of-anthropic-s-global-workspace-paper)
says J-Lens looks useful for **generating** hypotheses in model forensics, but its value
for **validating** them is unproven, and he flags a specific confound in the multi-hop
factual-recall setting: for "capital of the country that makes champagne -> Paris", maybe
`France` and `Paris` are just linearly related in unembedding space, so editing `France`
works only as a shortcut, not because the model does real multi-hop mediation. His own
27B replication stayed ambiguous for exactly this reason.

**We turn that ambiguity into a controlled test.** We edit the model's internal guess of a
hidden intermediate entity using each of three concept-direction methods, and measure
whether the final answer flips, as a function of how linearly readable the answer already
is from the entity.

## The claim ladder

- **C0 Detection** — J-Lens finds the hidden intermediate entity at middle/late layers,
  ranked above logit lens. (Existence.)
- **C1 Causal mediation** — steering the entity direction flips the answer, above a
  matched-norm random-direction control.
- **C2 Beats baselines** — J-Lens flips the answer more than logit-lens and a diff-of-means
  probe, at matched norm.
- **C3 Not an artifact (decisive)** — the J-Lens advantage **survives as `linsim -> 0`**,
  where `linsim = cos(U[answer], U[entity])`. If the advantage only exists at high `linsim`,
  J-Lens adds nothing over the unembedding. This is the exact confound Neel raised.

`linsim` is **measured per item**, not assumed, and is the x-axis of the headline figure.

## What J-Lens means here (honest scoping)

We implement the **single-token Jacobian** variant of J-Lens: the concept direction for
token `c` at layer `L` is `mean_pos d logit_c / d h_L`, averaged over a small corpus. This
is the cheap-to-bake version Neel names in his review ("even J-Lens computed from single
token Jacobians is better than logit lens in earlier layers"). It doubles as a reader
(dot with `h_L`) and a steering vector (add to `h_L`). Multi-token / future-token Jacobians
are the documented extension, not implemented here.

## Repo layout

```
data/prompts.json      multi-hop items with a HIGH->LOW linsim spread + counterfactuals
src/common.py          model load, tokenization, zero-shot verification, scoring
src/lenses.py          the 3 extractors (logit / J-Lens / diff-mean) + readers
src/experiment.py      detection sweep (C0) + causal steering with controls (C1/C2/C3)
src/plots.py           figures 1-4
src/run_all.py         checkpointed orchestrator + sanity-check summary
configs/smoke.yaml     Qwen3-0.6B, fits 8GB RAM  -> PIPELINE VALIDATION ONLY
configs/main.yaml      Qwen3-4B                  -> the real scientific run
modal_run.py           run main.yaml on a Modal A10G and pull artifacts back
results/               verify/detection/steering/summary json (committed)
figures/               fig1-4 png (committed)
checkpoints/           baked jvecs (gitignored; regenerated)
```

## How to run

Pipeline validation (any machine, ~8GB RAM). A 0.6B model largely cannot do these
recalls, so **effects are weak/noisy by design** — this only proves the harness runs:

```bash
pip install -r requirements.txt
python -m src.run_all --config configs/smoke.yaml
```

The real run (Qwen3-4B) on a GPU. With Modal authenticated:

```bash
python -m modal run modal_run.py            # runs configs/main.yaml on an A10G
```

Or on any CUDA box directly:

```bash
python -m src.run_all --config configs/main.yaml
```

Every stage checkpoints; re-running resumes. Use `--force` to recompute.

## Results (Qwen3-4B, n=17 kept items, L*=31, alpha=0.5; three lenses, held-out fitting, bootstrap CIs)

Headline: **J-Lens is a better *reader* of the hidden intermediate than logit lens, but as a
*causal lever* it is statistically indistinguishable from logit lens, and its apparent
mediation is mostly token-injection.** Read-vs-write split, now with baselines + stats.

- **C0 Detection (robust positive):** J-Lens MRR of the true hidden entity = **0.60** vs
  logit-lens **0.46** at L*=31, and J-Lens >= logit across most mid-late layers (`fig1`).
  The learned-linear **tuned lens** scores far lower (0.24) *but is undertrained* (fit on a
  30-sentence held-out corpus with a next-token objective), so it does NOT cleanly settle
  whether J-Lens' edge is unique to the Jacobian or shared by any downstream-aware map. A
  properly-trained tuned lens is the outstanding baseline. As-is, the trustworthy claim is
  **J-Lens > logit lens at detection**.
- **C1 Mediation (real but weak, and mostly token-push):** steering the J-Lens entity
  direction moves **0.036** probability onto the counterfactual answer (95% CI
  [0.007, 0.077], excludes 0) vs **-0.001** random. BUT the same steering raises the swapped
  *entity* token by **0.21** (`token_push` >> `toward_Ap`), and direct answer-steering
  dominates entity-steering **~23x** (0.84 vs 0.036). So the "mediation" is largely the model
  surfacing the injected entity token, not genuine multi-hop routing to the answer.
- **C2 (null vs baseline, now tested):** J-Lens (**0.039**) vs logit lens (**0.039**) are
  indistinguishable: paired permutation **p = 0.998**, 95% CI of the difference
  [-0.024, +0.019] straddles 0 (`fig2`). No causal advantage over the cheap baseline.
- **C3 (inconclusive):** the `(J-Lens - logit)` advantage vs linsim has slope **-0.11**, 95%
  CI [-0.27, +0.006] includes 0 (`fig3`). Even with low-linsim `symbol` items added, there
  is no significant linsim-dependence; the confound question is not resolved, just unrefuted.
- **`fig5`** shows the causal effect exists only at the gentlest alpha (0.5) and vanishes as
  larger alphas break coherence, so the headline uses alpha=0.5 by an a-priori rule.

**Honest limitations:** n=17 after the zero-shot filter (families kept: 7 capital, 7 currency,
3 symbol; the animal-sound family was dropped because the 4B wouldn't answer that phrasing).
The tuned-lens baseline is undertrained (above). Single-token J-Lens only; same-position
steering (which cannot fully separate injection from mediation — the token-push result is
that concern made visible). Sanity gate passed (random control ~0). Earlier-run pathologies
(suppression-gameable metric, over-large alpha) are documented in `results/_run1_*` + git log.

## Reading the results

`results/summary.json -> headline` reports, at the chosen layer `L*` and best steering
strength: C0 detection MRR (jlens vs logit), C1 concept effect vs random control, C2 jlens
vs logit, C3 low-`linsim` effects and the slope of `(jlens - logit)` effect vs `linsim`
(the decisive number), and the answer-swap dominance control.

Figures: `fig1_detection` (C0), `fig2_causal_vs_linsim` (C2/C3, the headline),
`fig3_confound` (C3 slope), `fig4_controls` (concept vs answer-swap vs random).

## Sanity checks you should do before trusting any of this

This is built so the numbers are checkable, not taken on faith:
- Read `results/verify.json` — confirm the model actually answers the kept items zero-shot.
- Read a handful of raw steering records in `results/steering.json` and confirm a "flip"
  is a real move toward the counterfactual answer, not the model just emitting the swapped
  entity token (`delta_Ip` should not dominate `delta_Ap`).
- Re-run with a different `seed` and confirm the random control stays near zero.

## Limitations

- Single-token J-Lens only (no future-token Jacobian).
- Detection/reading is over a candidate token set, not the full vocabulary.
- Steering injects at the last prompt token (same-position), not cross-position patching.
- Small item count; treat effect sizes as indicative, report uncertainty.
- The 0.6B smoke run is **not** a scientific result and must never be presented as one.
