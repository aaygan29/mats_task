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

## Results (Qwen3-4B, n=14 kept items, L*=27, alpha=0.5)

Headline: **J-Lens is a better *reader* of the hidden intermediate than logit lens, but
not a better *causal lever*.** This is the read-vs-write split Neel's review predicts
("J-Lens is very much about variable interpretability").

- **C0 Detection (robust positive):** J-Lens MRR of the true hidden entity = **0.81** vs
  logit-lens **0.47** at L*=27 (`fig1`). The advantage is concentrated at late layers; at
  mid layers single-token logit lens is comparable or better (consistent with Neel's note
  that the single-token J-Lens variant only mildly helps mid-stack).
- **C1 Mediation (real, partial):** steering the J-Lens entity direction moves **0.107**
  probability onto the counterfactual answer vs **0.0001** for a matched-norm random
  direction; 93% of trials flip. So the read is causally live, but direct answer-steering
  dominates entity-steering ~4.5x (**0.478** vs 0.107), i.e. the multi-hop mediation is weak.
- **C2/C3 (null vs baseline):** J-Lens (**0.107**) is statistically tied with logit lens
  (**0.114**), and the `(J-Lens - logit)` advantage is ~0 at every linsim (slope **+0.05**,
  intercept ~0; `fig2`, `fig3`). J-Lens' causal power is neither a low-linsim artifact nor
  an improvement over the cheap baseline.
- **Qualitatively (read the raw records):** clean genuine flips exist (`cur03` China->Britain
  moves yuan->pound, 0.76 mass, ~0 token-push), but some items are mostly token-push
  (`cap07`, `cap09`), and effects are heterogeneous.

**Honest limitations of this run:** n=14 after the zero-shot filter; the animal-sound family
(intended very-low-linsim anchor) was dropped because the 4B didn't answer those prompts in
top-6, so the lowest linsim is ~0 (currencies), not negative. Single-token J-Lens only;
same-position steering; small N. Earlier runs' pathologies (a suppression-gameable metric,
over-large alpha) are documented in `results/_run1_qwen3-4B_INVALID_causal/` and the git log.

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
